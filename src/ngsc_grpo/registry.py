from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    directory: str
    modality: str
    foreground_classes: tuple[str, ...]
    normal_class: str
    normal_prompt_key: str
    abnormal_prompt_key: str
    radiology: bool
    label_kind: str
    patient_strategy: str = "image_level"


DATASETS: Dict[str, DatasetSpec] = {
    "BrainMRI": DatasetSpec(
        "BrainMRI", "brain_tumors", "MRI",
        ("pituitary tumor", "meningioma tumor", "glioma tumor"),
        "normal brain", "normal brain", "abnormal brain", True, "brain_prompt_map"
    ),
    "BUSI": DatasetSpec(
        "BUSI", "breast_tumors", "Ultrasound",
        ("benign tumor", "malignant tumor"),
        "normal scan", "normal scan", "breast_abnormal", True, "breast_prompt_map"
    ),
    "KiTS": DatasetSpec(
        "KiTS", "kits_2d", "CT", ("kidney tumor",),
        "normal kidney", "normal kidney", "abnormal kidney", True, "kits"
    ),
    "ColonDB": DatasetSpec(
        "ColonDB", "colondb_polyp", "Endoscopy", ("polyp",),
        "normal endoscopy", "normal endoscopy", "abnormal endoscopy", False, "same_name"
    ),
    "Covid-QU-Ex": DatasetSpec(
        "Covid-QU-Ex", "Covid-Qu-Ex", "X-ray", ("covid lungs",),
        "normal lungs", "normal lungs", "abnormal lungs", True, "same_name"
    ),
    "MedSeg": DatasetSpec(
        "MedSeg", "medseg_covid", "CT",
        ("ground_glass", "consolidation", "pleural_effusion"),
        "normal lungs CT", "normal lungs CT", "abnormal lungs CT", True, "medseg"
    ),
    "HAM10000": DatasetSpec(
        "HAM10000", "ham10000", "Dermoscopy",
        (
            "actinic keratoses", "basal cell carcinoma", "benign keratosis-like lesions",
            "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions"
        ),
        "normal skin", "normal skin", "abnormal skin", False, "ham10000", "lesion_id"
    ),
    "PH2": DatasetSpec(
        "PH2", "ph2", "Dermoscopy",
        ("common nevus", "atypical nevus", "melanoma"),
        "normal skin", "normal skin", "abnormal skin", False, "ph2"
    ),
}


HAM_DX = {
    "akiec": "actinic keratoses",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesions",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevi",
    "vasc": "vascular lesions",
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset {name!r}; choices={tuple(DATASETS)}") from exc


def normalize_class_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _load_prompt_map(prompt_map_dir: Path, filename: str) -> Dict[str, str]:
    with (prompt_map_dir / filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _active_from_sentence(sentence: str, spec: DatasetSpec) -> str:
    value = sentence.lower()
    for class_name in spec.foreground_classes:
        token = class_name.replace(" tumor", "")
        if token in value:
            return class_name
    raise ValueError(f"Could not infer {spec.name} class from prompt sentence: {sentence}")


def _ham_metadata(dataset_dir: Path) -> Dict[str, Mapping[str, str]]:
    path = dataset_dir / "HAM10000_metadata.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {f"{row['image_id'].strip()}.jpg": row for row in rows}


def _ph2_active(dataset_dir: Path, image_stem: str) -> str:
    mapping = {
        "common nevus": "common_nevus",
        "atypical nevus": "atypical_nevus",
        "melanoma": "melanoma",
    }
    found = [
        name for name, folder in mapping.items()
        if (dataset_dir / "test_masks" / folder / f"{image_stem}_lesion.bmp").is_file()
    ]
    if len(found) != 1:
        raise ValueError(f"Expected one PH2 class mask for {image_stem}, found {found}")
    return found[0]


def _medseg_present(dataset_dir: Path, image_name: str) -> List[str]:
    stem = Path(image_name).stem
    mask_name = f"msk{stem[2:]}{Path(image_name).suffix}" if stem.startswith("im") else image_name
    present: List[str] = []
    for class_name in ("ground_glass", "consolidation", "pleural_effusion"):
        path = dataset_dir / "test_masks" / class_name / mask_name
        if path.is_file() and np.asarray(Image.open(path).convert("L")).max() > 0:
            present.append(class_name)
    return present


def _single_mask_path(spec: DatasetSpec, dataset_dir: Path, image_name: str) -> Path:
    if spec.label_kind == "kits":
        return dataset_dir / "test_masks" / f"{Path(image_name).stem.replace('_img', '_seg')}.png"
    if spec.label_kind == "ham10000":
        return dataset_dir / "test_masks" / f"{Path(image_name).stem}_segmentation.png"
    return dataset_dir / "test_masks" / image_name


def discover_records(
    data_root: Path,
    prompt_map_dir: Path,
    dataset_name: str,
    include_labels: bool,
) -> List[dict]:
    spec = get_spec(dataset_name)
    dataset_dir = data_root / spec.directory
    image_dir = dataset_dir / "test_images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    prompt_map = None
    ham_meta = None
    if include_labels and spec.label_kind == "brain_prompt_map":
        prompt_map = _load_prompt_map(prompt_map_dir, "brain_tumors_testing.json")
    elif include_labels and spec.label_kind == "breast_prompt_map":
        prompt_map = _load_prompt_map(prompt_map_dir, "breast_tumors_testing.json")
    elif include_labels and spec.label_kind == "ham10000":
        ham_meta = _ham_metadata(dataset_dir)

    records: List[dict] = []
    for image_path in image_paths:
        present: List[str] = []
        patient_id = image_path.stem
        if include_labels:
            if prompt_map is not None:
                if image_path.name not in prompt_map:
                    raise KeyError(f"Missing prompt-map label for {image_path.name}")
                present = [_active_from_sentence(prompt_map[image_path.name], spec)]
            elif spec.label_kind in {"same_name", "kits"}:
                mask_path = _single_mask_path(spec, dataset_dir, image_path.name)
                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                if np.asarray(Image.open(mask_path).convert("L")).max() > 0:
                    present = [spec.foreground_classes[0]]
            elif spec.label_kind == "medseg":
                present = _medseg_present(dataset_dir, image_path.name)
            elif spec.label_kind == "ham10000":
                row = ham_meta.get(image_path.name)
                if row is None:
                    raise KeyError(f"Missing HAM10000 metadata for {image_path.name}")
                present = [HAM_DX[row["dx"].strip().lower()]]
                patient_id = row.get("lesion_id", image_path.stem).strip() or image_path.stem
            elif spec.label_kind == "ph2":
                present = [_ph2_active(dataset_dir, image_path.stem)]
            else:
                raise NotImplementedError(spec.label_kind)

        records.append(
            {
                "dataset": spec.name,
                "modality": spec.modality,
                "image_id": image_path.name,
                "patient_id": patient_id,
                "image_relpath": str(image_path.relative_to(data_root)),
                "present_classes": present,
                "prompt_classes": list(spec.foreground_classes),
                "patient_strategy": spec.patient_strategy,
            }
        )
    return records


def load_class_masks(data_root: Path, row: Mapping[str, str], size=None) -> Dict[str, np.ndarray]:
    spec = get_spec(row["dataset"])
    dataset_dir = data_root / spec.directory
    image_name = row["image_id"]
    image = Image.open(data_root / row["image_relpath"])
    width, height = image.size

    def read_mask(path: Path) -> np.ndarray:
        if not path.is_file():
            return np.zeros((height, width), dtype=bool)
        mask_image = Image.open(path).convert("L")
        if mask_image.size != (width, height):
            mask_image = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
        return np.asarray(mask_image) > 0

    masks = {name: np.zeros((height, width), dtype=bool) for name in spec.foreground_classes}
    if spec.label_kind in {"brain_prompt_map", "breast_prompt_map"}:
        prompt_file = "brain_tumors_testing.json" if spec.label_kind.startswith("brain") else "breast_tumors_testing.json"
        prompt_map_dir = data_root.parent / "assets" / "prompt_maps"
        prompt_map = _load_prompt_map(prompt_map_dir, prompt_file)
        active = _active_from_sentence(prompt_map[image_name], spec)
        masks[active] = read_mask(dataset_dir / "test_masks" / image_name)
    elif spec.label_kind in {"same_name", "kits"}:
        masks[spec.foreground_classes[0]] = read_mask(_single_mask_path(spec, dataset_dir, image_name))
    elif spec.label_kind == "medseg":
        stem = Path(image_name).stem
        mask_name = f"msk{stem[2:]}{Path(image_name).suffix}" if stem.startswith("im") else image_name
        for name in spec.foreground_classes:
            masks[name] = read_mask(dataset_dir / "test_masks" / name / mask_name)
    elif spec.label_kind == "ham10000":
        row_meta = _ham_metadata(dataset_dir)[image_name]
        active = HAM_DX[row_meta["dx"].strip().lower()]
        masks[active] = read_mask(_single_mask_path(spec, dataset_dir, image_name))
    elif spec.label_kind == "ph2":
        mapping = {
            "common nevus": "common_nevus",
            "atypical nevus": "atypical_nevus",
            "melanoma": "melanoma",
        }
        for name, folder in mapping.items():
            masks[name] = read_mask(dataset_dir / "test_masks" / folder / f"{Path(image_name).stem}_lesion.bmp")
    else:
        raise NotImplementedError(spec.label_kind)

    if size is not None:
        out = {}
        target_h, target_w = int(size[0]), int(size[1])
        for name, mask in masks.items():
            resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (target_w, target_h), resample=Image.Resampling.NEAREST
            )
            out[name] = np.asarray(resized) > 0
        masks = out
    return masks


def prompt_inventory(dataset_names: Iterable[str]) -> Dict[str, List[str]]:
    return {name: list(get_spec(name).foreground_classes) for name in dataset_names}

