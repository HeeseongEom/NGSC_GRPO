import numpy as np

from ngsc_grpo.evaluation import _assemble_multiclass, _binary_auroc, _dice, _safe_iou


def test_binary_metrics():
    gt = np.array([[0, 0], [1, 1]], dtype=bool)
    pred = np.array([[0, 1], [1, 1]], dtype=bool)
    assert np.isclose(_safe_iou(pred, gt), 2 / 3)
    assert np.isclose(_dice(pred, gt), 4 / 5)
    assert _binary_auroc(np.array([[0.0, 0.1], [0.9, 1.0]]), gt) == 1.0


def test_empty_ground_truth_is_not_counted_as_positive_iou():
    empty = np.zeros((2, 2), dtype=bool)
    assert np.isnan(_safe_iou(empty, empty))
    assert np.isnan(_dice(empty, empty))


def test_multiclass_assembly_is_mutually_exclusive_and_respects_class_thresholds():
    scores = np.array(
        [
            [[0.8, 0.7], [0.1, 0.9]],
            [[0.9, 0.6], [0.8, 0.95]],
        ]
    )
    masks = _assemble_multiclass(scores, np.array([0.75, 0.85]))
    assert masks[1][0, 0] and not masks[0][0, 0]  # both eligible, higher score wins
    assert not masks[0][0, 1] and not masks[1][0, 1]  # neither eligible -> background
    assert not masks[0][1, 1]
    assert masks[1][1, 1]
    assert not np.logical_and(masks[0], masks[1]).any()
