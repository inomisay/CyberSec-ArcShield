"""Combine attack metrics and benign BRR reports into security-usability tables.

Example:
  python -m src.evaluation.compare_security_usability --attack-metrics output/model_results/<attack_run>/.../combined/metrics.json --brr-report output/model_results/<benign_run>/.../combined/brr_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.analysis.metrics import security_usability_tradeoff


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_by_system(items: list[dict[str, Any]], system_field: str) -> dict[str, dict[str, Any]]:
    return {str(item.get(system_field, "overall")): item for item in items}


def comparison_rows(attack_metrics: dict[str, Any], brr_report: dict[str, Any]) -> list[dict[str, Any]]:
    attack_by_condition = rows_by_system(attack_metrics.get("per_condition", attack_metrics.get("per_defense_configuration", [])), "defense_config")
    brr_by_condition = rows_by_system(brr_report.get("per_defense_layer", []), "defense_config")
    systems = sorted(set(attack_by_condition) | set(brr_by_condition))
    rows = []
    for system in systems:
        attack_row = attack_by_condition.get(system, {})
        brr_row = brr_by_condition.get(system, {})
        asr = float(attack_row.get("asr", 0.0))
        dsr = float(attack_row.get("dsr", 0.0))
        brr = float(brr_row.get("benign_refusal_rate", 0.0))
        rows.append({"system": system, **security_usability_tradeoff(asr, dsr, brr)})
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ArcShield Security-Usability Comparison",
        "",
        "| System | ASR lower is better | DSR higher is better | BRR lower is better | Defense Aggressiveness |",
        "| ------ | -----: | -----: | -----: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['asr']:.4f} | {row['dsr']:.4f} | "
            f"{row['brr']:.4f} | {row['defense_aggressiveness']} |"
        )
    return "\n".join(lines)


def export_bar_chart(rows: list[dict[str, Any]], path: Path) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    labels = [row["system"] for row in rows]
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 1.2), 4.4), dpi=220)
    width = 0.25
    ax.bar([x - width for x in positions], [row["asr"] * 100.0 for row in rows], width=width, label="ASR (%)", color="#b91c1c")
    ax.bar(positions, [row["dsr"] * 100.0 for row in rows], width=width, label="DSR (%)", color="#166534")
    ax.bar([x + width for x in positions], [row["brr"] * 100.0 for row in rows], width=width, label="BRR (%)", color="#2563eb")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def export_tradeoff_plot(rows: list[dict[str, Any]], path: Path) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=220)
    for row in rows:
        ax.scatter(row["brr"] * 100.0, row["asr"] * 100.0, s=80)
        ax.annotate(row["system"], (row["brr"] * 100.0, row["asr"] * 100.0), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("BRR (%) lower is better")
    ax.set_ylabel("ASR (%) lower is better")
    ax.set_title("Security vs Usability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = comparison_rows(load_json(args.attack_metrics), load_json(args.brr_report))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "security_usability_comparison.json"
    table_path = output_dir / "security_usability_comparison.md"
    chart_path = output_dir / "security_usability_comparison.png"
    tradeoff_path = output_dir / "security_vs_usability_plot.png"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table_path.write_text(markdown_table(rows), encoding="utf-8")
    chart_result = export_bar_chart(rows, chart_path)
    tradeoff_result = export_tradeoff_plot(rows, tradeoff_path)
    return {
        "json": str(json_path),
        "table": str(table_path),
        "chart": chart_result,
        "security_vs_usability_plot": tradeoff_result,
        "rows": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ASR/DSR/BRR security-usability comparison artifacts.")
    parser.add_argument("--attack-metrics", required=True)
    parser.add_argument("--brr-report", required=True)
    parser.add_argument("--output-dir", default="output/model_results/security_usability_comparison")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
