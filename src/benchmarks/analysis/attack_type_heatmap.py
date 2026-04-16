from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_attack_type_heatmap(output_path: Path | None = None) -> Path:
    data = pd.DataFrame(
        {
            "Context Camouflage": [3, 5, 3, 4],
            "Policy Bypass Likelihood": [4, 4, 4, 4],
            "Reasoning Exploitation": [2, 2, 5, 3],
            "Overall Risk": [4, 4, 5, 4],
        },
        index=[
            "Pretending / Roleplay",
            "Attention Shifting",
            "Logic Trap",
            "Hypothetical Manipulation",
        ],
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 5.2))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        linewidths=0.7,
        linecolor="white",
        vmin=1,
        vmax=5,
        cbar_kws={"label": "Severity (1=Low, 5=High)"},
    )

    ax.set_title("Prompt Attack Type Risk Heatmap", fontsize=14, pad=12)
    ax.set_xlabel("Evaluation Dimension")
    ax.set_ylabel("Attack Type")
    plt.tight_layout()

    if output_path is None:
        output_path = Path("logs") / "attack_type_heatmap.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=240)
    plt.close()
    return output_path


if __name__ == "__main__":
    path = build_attack_type_heatmap()
    print(f"Heatmap saved to: {path.resolve()}")
