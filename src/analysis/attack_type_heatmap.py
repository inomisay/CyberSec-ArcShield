"""Build heatmaps from the curated ArcShield attack dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "curated" / "attack_dataset_curated.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "attack_types"


ATTACK_TYPE_ORDER = [
    "Direct Exploitation",
    "Jailbreaks",
    "Prompt Injection",
    "Indirect Injection",
    "Role Playing / Pretending",
    "Attention Shifting",
    "Logic Trap Attacks",
    "Hypothetical Manipulation",
    "Accidental Context Leakage",
    "Code Injection",
    "Dictionary Attack",
    "Fill-in-the-Blank Attack",
    "Multi-Language Attack",
    "Multi-Prompt Attack",
    "Obfuscation (Token Smuggling)",
    "Payload Splitting",
]


def load_curated_dataset(dataset_path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Curated dataset not found: {path}")

    df = pd.read_csv(path)
    required = {"prompt", "attack_type", "source_dataset", "is_synthetic"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["is_synthetic"] = df["is_synthetic"].astype(str).str.lower().isin({"true", "1", "yes"})
    return df


def attack_type_source_matrix(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    matrix = pd.crosstab(df["attack_type"], df["source_dataset"])
    matrix = matrix.reindex([name for name in ATTACK_TYPE_ORDER if name in matrix.index])
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    if normalize:
        matrix = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0) * 100.0
    return matrix


def attack_type_synthetic_matrix(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    labeled = df.copy()
    labeled["prompt_origin"] = labeled["is_synthetic"].map({True: "Synthetic", False: "Real"})
    matrix = pd.crosstab(labeled["attack_type"], labeled["prompt_origin"])
    matrix = matrix.reindex([name for name in ATTACK_TYPE_ORDER if name in matrix.index])
    matrix = matrix.reindex(["Real", "Synthetic"], axis=1, fill_value=0)
    if normalize:
        matrix = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0) * 100.0
    return matrix


def save_heatmap(
    matrix: pd.DataFrame,
    output_path: str | Path,
    title: str,
    normalize: bool = False,
    figsize: tuple[float, float] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if figsize is None:
        figsize = (max(10.0, matrix.shape[1] * 0.72), max(7.0, matrix.shape[0] * 0.42))

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=figsize, dpi=240)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.45,
        linecolor="white",
        annot=True,
        fmt=".1f" if normalize else ".0f",
        cbar_kws={"label": "Row %" if normalize else "Prompt count"},
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(matrix.columns.name or "")
    ax.set_ylabel(matrix.index.name or "attack_type")
    ax.tick_params(axis="x", labelrotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def build_attack_type_heatmap(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    view: str = "source",
    normalize: bool = False,
) -> dict[str, Path]:
    df = load_curated_dataset(dataset_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if view == "source":
        matrix = attack_type_source_matrix(df, normalize=normalize)
        stem = "attack_type_by_source_heatmap_pct" if normalize else "attack_type_by_source_heatmap"
        title = "ArcShield Attack Types by Source Dataset"
    elif view == "synthetic":
        matrix = attack_type_synthetic_matrix(df, normalize=normalize)
        stem = "attack_type_real_synthetic_heatmap_pct" if normalize else "attack_type_real_synthetic_heatmap"
        title = "ArcShield Attack Types: Real vs Synthetic Prompts"
    else:
        raise ValueError("view must be 'source' or 'synthetic'")

    png_path = output / f"{stem}.png"
    csv_path = output / f"{stem}.csv"
    matrix.to_csv(csv_path)
    save_heatmap(matrix, png_path, title=title, normalize=normalize)
    return {"png": png_path, "csv": csv_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ArcShield attack-type heatmaps from the curated dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to attack_dataset_curated.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for PNG/CSV outputs")
    parser.add_argument("--view", choices=["source", "synthetic", "both"], default="source")
    parser.add_argument("--normalize", action="store_true", help="Plot row percentages instead of raw counts")
    args = parser.parse_args()

    views = ["source", "synthetic"] if args.view == "both" else [args.view]
    for view in views:
        paths = build_attack_type_heatmap(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            view=view,
            normalize=args.normalize,
        )
        print(f"{view} heatmap saved: {paths['png']}")
        print(f"{view} matrix saved:  {paths['csv']}")


if __name__ == "__main__":
    main()
