import argparse
import csv
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from compare_summary import BENCHMARKS, TASKS, parse_score
from summarize_recent_json import (
    COLUMNS as RUN_COLUMNS,
    SCORE_COLUMNS,
    build_run_row,
    get_run_timestamp,
    load_json,
    non_negative_int,
)


COLUMNS = ["Temperature", *[column for column in RUN_COLUMNS if column != "Timestamp"]]


def get_args():
    parser = argparse.ArgumentParser(description="Summarize and plot results by temperature.")
    parser.add_argument("-d", "--dir", required=True, help="Directory prefix containing benchmark results.")
    parser.add_argument("-t", "--type", default="naive", choices=["naive", "cot", "few_shot", "custom"])
    parser.add_argument("-m", "--models", nargs="+", default=["qwen3:4b-instruct", "gpt-6.1-sol"])
    parser.add_argument("-k", "--keep", type=non_negative_int, default=0, help="Recent runs to keep per model; 0 keeps all.")
    parser.add_argument("-o", "--output", default="temperature_summary.tsv", help="Output TSV file.")
    parser.add_argument("-p", "--plot", help="Output plot; defaults to the TSV name with a .png suffix.")
    return parser.parse_args()


def get_temperature(data, path):
    client = data.get("client", {})
    value = (client.get("generate_kwargs") or {}).get("temperature", client.get("temperature"))
    if value is None:
        raise ValueError(f"Missing client temperature in {path}")
    return float(value)


def average_cells(cells):
    """Average a list of score cells, returning a formatted string with the average and standard error."""
    scores = [parse_score(cell) for cell in cells if cell]
    if not scores:
        return ""

    average = sum(score for score, _ in scores) / len(scores)
    errors = [error for _, error in scores]
    if any(error is None for error in errors):
        return f"{average:.2f}"
    error = sqrt(sum(value ** 2 for value in errors)) / len(errors)
    return f"{average:.2f}±{error:.2f}"


def summarize(input_dir, summary_type, models, keep):
    input_dir = Path(input_dir)
    result_dirs = [path for path in input_dir.parent.glob(f"{input_dir.name}*") if path.is_dir()]
    if not result_dirs:
        raise ValueError(f"No results directories match: {input_dir}*")

    # Group runs by model and temperature, keeping only the most recent runs if specified
    runs_by_model = {}
    for path in sorted(file for directory in result_dirs for file in directory.rglob("summary.json")):
        data = load_json(path)
        model = data.get("client", {}).get("model_id")
        if data.get("agent", {}).get("type") != summary_type or model not in models:
            continue
        row = build_run_row(path, data, model)
        row["Temperature"] = get_temperature(data, path)
        runs_by_model.setdefault(model, []).append((get_run_timestamp(path), row))

    groups = {}
    selected_runs = []
    for runs in runs_by_model.values():
        runs.sort(reverse=True, key=lambda item: item[0])
        for _, row in runs[:keep] if keep else runs:
            groups.setdefault((row["Temperature"], row["Model"]), []).append(row)
            selected_runs.append(row)

    rows = []
    for (temperature, model), group in groups.items():
        row = {"Temperature": temperature, "Model": model}
        for column in SCORE_COLUMNS:
            row[column] = average_cells([run.get(column, "") for run in group])
        rows.append(row)
    rows.sort(key=lambda row: (row["Temperature"], row["Model"]))
    return rows, selected_runs


def write_table(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def point_positions(center, count, width):
    if count <= 1:
        return [center]
    return [center + width * (index / (count - 1) - 0.5) for index in range(count)]


def plot_by_temperature(runs, output):
    temperatures = sorted({row["Temperature"] for row in runs})
    models = list(dict.fromkeys(row["Model"] for row in runs))
    colors = {value: plt.get_cmap("tab10")(index) for index, value in enumerate(temperatures)}
    task_names = [task for benchmark in BENCHMARKS for task in TASKS[benchmark]]
    markers = {task: marker for task, marker in zip(task_names, ["s", "^", "D", "P", "X", "v"])}

    figure, axes = plt.subplots(
        1, len(models), figsize=(max(8, 5 * len(models)), 5), sharey=True, squeeze=False
    )
    box_width = 0.75 / len(temperatures)

    for axis, model in zip(axes[0], models):
        model_runs = [row for row in runs if row["Model"] == model]
        for benchmark_index, benchmark in enumerate(BENCHMARKS):
            for temperature_index, temperature in enumerate(temperatures):
                group = [row for row in model_runs if row["Temperature"] == temperature]
                position = benchmark_index + (temperature_index - (len(temperatures) - 1) / 2) * box_width
                scores = [parse_score(row.get(benchmark))[0] for row in group]
                scores = [score for score in scores if score is not None]
                if scores:
                    axis.boxplot(
                        scores,
                        positions=[position],
                        widths=box_width * 0.75,
                        patch_artist=True,
                        manage_ticks=False,
                        showfliers=False,
                        boxprops={"facecolor": colors[temperature], "alpha": 0.45},
                        medianprops={"color": "black"},
                    )
                    axis.scatter(
                        point_positions(position, len(scores), box_width * 0.3),
                        scores,
                        color=colors[temperature],
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=3,
                    )

                for task_index, task in enumerate(TASKS[benchmark]):
                    task_scores = [parse_score(row.get(task))[0] for row in group]
                    task_scores = [score for score in task_scores if score is not None]
                    if task_scores:
                        task_position = position + (task_index - 0.5) * box_width * 0.2
                        axis.scatter(
                            point_positions(task_position, len(task_scores), box_width * 0.2),
                            task_scores,
                            color=colors[temperature],
                            marker=markers[task],
                            edgecolors="black",
                            linewidths=0.5,
                            zorder=4,
                        )

        axis.set_title(model)
        axis.set_xticks(range(len(BENCHMARKS)), BENCHMARKS)
        axis.set_xlabel("Benchmark")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)

    axes[0][0].set_ylabel("Score (%)")
    handles = [Patch(facecolor=colors[value], alpha=0.6, label=f"temperature={value:g}") for value in temperatures]
    handles += [
        Line2D([0], [0], marker=markers[task], color="none", markerfacecolor="gray",
               markeredgecolor="black", label=task)
        for task in task_names
    ]
    figure.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, 1.02))
    figure.tight_layout(rect=(0, 0, 1, 0.82))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = get_args()
    rows, runs = summarize(args.dir, args.type, args.models, args.keep)
    if not rows:
        raise SystemExit(f"No {args.type!r} results found in {args.dir}")

    output = write_table(rows, args.output)
    plot = Path(args.plot) if args.plot else output.with_suffix(".png")
    plot_by_temperature(runs, plot)
    print(f"Saved summary to {output}")
    print(f"Saved plot to {plot}")


if __name__ == "__main__":
    main()
