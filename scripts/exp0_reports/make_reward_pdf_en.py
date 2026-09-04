#!/usr/bin/env python3
"""Render an implementation-explicit EXP0 reward definition."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "ngsc_grpo_exp0" / "reports" / "EXP0_reward_function_en.pdf"
HEADING_FONT = "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf"
BODY_FONT = "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    heading = FontProperties(fname=HEADING_FONT)
    body = FontProperties(fname=BODY_FONT)
    mpl.rcParams["mathtext.fontset"] = "stix"

    with PdfPages(OUTPUT, metadata={
        "Title": "EXP0 Reward",
        "Author": "NGSC-GRPO",
        "Subject": "Implementation-explicit reward used for Global and CNN GRPO training",
    }) as pdf:
        fig = plt.figure(figsize=(11, 8.5), facecolor="white")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.axis("off")

        blue = "#4F81BD"
        black = "#111111"
        gray = "#B7B7B7"
        left, right = 0.068, 0.942

        fig.text(left, 0.900, "Reward", fontproperties=heading,
                 fontsize=17, color=blue)

        fig.text(left, 0.840, "Binary prediction", fontproperties=body,
                 fontsize=13, color=black)
        fig.text(0.260, 0.840,
                 r"$\widehat{M}_{a,ij}=\mathbf{1}\!\left[s_a(i,j)\geq\eta_a\right],"
                 r"\qquad i=1,\ldots,H,\quad j=1,\ldots,W$",
                 fontsize=16, color=black)
        fig.add_artist(plt.Line2D((left, right), (0.805, 0.805),
                                 color=black, linewidth=0.65))

        fig.text(left, 0.750, "Ground-truth condition", fontproperties=body,
                 fontsize=13, color=black)
        fig.text(0.300, 0.750, "Reward used for the sample", fontproperties=body,
                 fontsize=13, color=black)

        fig.text(left, 0.665, "Foreground present", fontproperties=body,
                 fontsize=13, color=black)
        fig.text(left, 0.625, r"$\sum_{i=1}^{H}\sum_{j=1}^{W}Y_{ij}>0$",
                 fontsize=14, color=black)
        fig.text(0.300, 0.645,
                 r"$R(a;Y)=\dfrac{2\sum_{i=1}^{H}\sum_{j=1}^{W}"
                 r"\widehat{M}_{a,ij}Y_{ij}}"
                 r"{\sum_{i=1}^{H}\sum_{j=1}^{W}\widehat{M}_{a,ij}"
                 r"+\sum_{i=1}^{H}\sum_{j=1}^{W}Y_{ij}}$",
                 fontsize=17, color=black)

        fig.text(left, 0.515, "Foreground absent", fontproperties=body,
                 fontsize=13, color=black)
        fig.text(left, 0.475, r"$\sum_{i=1}^{H}\sum_{j=1}^{W}Y_{ij}=0$",
                 fontsize=14, color=black)
        fig.text(0.300, 0.495,
                 r"$R(a;Y)=1-\dfrac{1}{H\,W}"
                 r"\sum_{i=1}^{H}\sum_{j=1}^{W}\widehat{M}_{a,ij}$",
                 fontsize=17, color=black)

        fig.add_artist(plt.Line2D((left, right), (0.420, 0.420),
                                 color=gray, linewidth=0.45))
        fig.text(left, 0.370, "Symbols", fontproperties=heading,
                 fontsize=14, color=blue)

        definitions = [
            (r"$a=(\eta,\tau,\gamma,\kappa_{sp})$", "sampled controller action"),
            (r"$s_a(i,j)$", "upsampled refined foreground score after applying action a"),
            (r"$\eta_a$", "foreground threshold contained in action a"),
            (r"$Y_{ij}\in\{0,1\}$", "ground-truth label at pixel (i, j)"),
            (r"$\widehat{M}_{a,ij}\in\{0,1\}$", "predicted foreground label at pixel (i, j)"),
            (r"$H,W$", "reward-map height and width; H = W = 224 in EXP0"),
            (r"$H\,W$", "total number of pixels in the 224 x 224 reward map"),
        ]
        y = 0.320
        for symbol, meaning in definitions:
            fig.text(left + 0.025, y, symbol, fontsize=12.7, color=black)
            fig.text(0.300, y, meaning, fontproperties=body, fontsize=12.3, color=black)
            y -= 0.041

        fig.text(left, 0.027,
                 "Exactly one branch is used per sample. The two rewards are not added, and 0 <= R <= 1.",
                 fontproperties=body, fontsize=11.5, color=black)

        pdf.savefig(fig)
        plt.close(fig)

    print(OUTPUT)


if __name__ == "__main__":
    main()
