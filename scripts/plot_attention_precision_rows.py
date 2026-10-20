"""Render per-row attention gradient error distributions from a precision report."""

import argparse
import json
import math
from pathlib import Path


def distribution(row, gradient):
    data = row["row_metrics"][gradient]
    values = [value for value in data["log10_relative_error"] if value is not None and math.isfinite(value)]
    zeros = sum(
        value == 0 and log_value is None for value, log_value in zip(data["relative_error"], data["log10_relative_error"])
    )
    # A finite log error still describes ratios that overflow the relative-error field.
    undefined = sum(
        value is None and log_value is None for value, log_value in zip(data["relative_error"], data["log10_relative_error"])
    )
    return values, zeros, undefined


def render(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in report["rows"] if row["status"] == "ok"]
    if not rows or any("row_metrics" not in row for row in rows):
        raise ValueError(
            "report needs successful rows with row_metrics; rerun scripts.benchmark_attention_precision with --row-errors"
        )
    scales = sorted({row["requested_scale"] or row["max_absolute_logit"] for row in rows})
    backends = list(dict.fromkeys(row["backend"] for row in rows))
    figure, axes = plt.subplots(len(scales), 3, figsize=(12, 3.4 * len(scales)), squeeze=False, sharey="row")
    try:
        for row_index, scale in enumerate(scales):
            for column, gradient in enumerate(("dq", "dk", "dv")):
                axis = axes[row_index, column]
                labels = []
                for position, backend in enumerate(backends):
                    selected = [
                        row
                        for row in rows
                        if row["backend"] == backend and (row["requested_scale"] or row["max_absolute_logit"]) == scale
                    ]
                    values, zeros, undefined = [], 0, 0
                    for row in selected:
                        samples, exact, missing = distribution(row, gradient)
                        values.extend(samples)
                        zeros += exact
                        undefined += missing
                    labels.append(f"{backend}\nn={len(values)}; zero={zeros}; undef={undefined}")
                    if values:
                        if len(values) > 1 and min(values) < max(values):
                            axis.violinplot([values], positions=[position], showmedians=True, showextrema=False)
                        else:
                            axis.scatter([position], [values[0]], marker="_")
                axis.set_xticks(range(len(backends)), labels, fontsize=7)
                axis.set_title(f"{gradient} / requested scale {scale:g}")
                axis.grid(axis="y", alpha=0.2)
                if column == 0:
                    axis.set_ylabel("log10 per-row relative L2 error")
        figure.suptitle(f"Attention gradient error by row / {report.get('device', 'unknown device')}")
        figure.text(
            0.5,
            0.01,
            "Zero-error and undefined rows are counted below each backend and omitted from the violins.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.96))
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, metadata={"Creator": "SimpleTuner attention precision", "Date": None})
    finally:
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        render(json.loads(args.input.read_text()), args.output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
