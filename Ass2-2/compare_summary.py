import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLUMNS = [
    "Model",
    "Average Progress",
    "BabyAI",
    "TextWorld",
    "BabaIsAI",
    "BabyAI-goto",
    "BabyAI-putnext",
    "TextWorld-coin_collector",
    "TextWorld-treasure_hunter",
    "BabaIsAI-goto_win",
    "BabaIsAI-make_win",
]

BENCHMARKS = ["BabyAI", "TextWorld", "BabaIsAI"]
TASKS = {
    "BabyAI": ["BabyAI-goto", "BabyAI-putnext"],
    "TextWorld": ["TextWorld-coin_collector", "TextWorld-treasure_hunter"],
    "BabaIsAI": ["BabaIsAI-goto_win", "BabaIsAI-make_win"],
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, nargs="+", help="Path to the input file containing evaluation results.")
    parser.add_argument("-l", "--labels", type=str, required=True, nargs="+", help="Labels for the input files (must match the number of input files).")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file for the plot.")
    return parser.parse_args()


def parse_score(value):
    # A cell looks like "95.0±4.87 (16.35)". The value in brackets is the average number of steps and is not part of this plot
    value = (value or "").split("(", 1)[0].strip()
    if not value:
        return None, None

    score, separator, error = value.partition("±")
    return float(score), float(error) if separator and error.strip() else None


def load_results(input_files):
    results = []
    for input_file in input_files:
        with Path(input_file).open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
        if not rows:
            raise ValueError(f"No result rows found in {input_file}")

        by_model = {}
        for row in rows:
            model = row.get("Model")
            if not model:
                raise ValueError(f"Missing Model value in {input_file}")
            by_model[model] = row
        results.append(by_model)

    return results


def plot_results(input_files, labels, output_file):
    results = load_results(input_files)

    # Keep the order from the TSV files, but include a model that only occurs in a later file as well
    model_names = []
    for rows in results:
        for model in rows:
            if model not in model_names:
                model_names.append(model)

    color_map = {
        label: plt.get_cmap("tab10")(index % 10)
        for index, label in enumerate(labels)
    }
    task_markers = ["o", "s", "^", "D", "P", "X"]
    marker_map = {
        task: task_markers[index % len(task_markers)]
        for index, task in enumerate(task for benchmark in BENCHMARKS for task in TASKS[benchmark])
    }

    figure, axes = plt.subplots(1, len(model_names), 
                                figsize=(max(8, 4 * len(model_names)), 5), sharey=True, squeeze=False)
    axes = axes[0]

    # Each subplot is one model
    # Within a benchmark, the bars compare Naive and Custom
    for axis, model in zip(axes, model_names):
        bar_width = 0.8 / len(labels)
        for label_index, (label, rows) in enumerate(zip(labels, results)):
            if model not in rows:
                raise ValueError(f"Model {model!r} is missing from the {label!r} results")

            row = rows[model]
            positions = [
                benchmark_index + (label_index - (len(labels) - 1) / 2) * bar_width
                for benchmark_index in range(len(BENCHMARKS))
            ]
            # Get the scores and errors for each benchmark, defaulting to 0.0 if missing
            scores_and_errors = [parse_score(row.get(benchmark)) for benchmark in BENCHMARKS]
            scores = [score if score is not None else 0.0 for score, _ in scores_and_errors]
            errors = [error if error is not None else 0.0 for _, error in scores_and_errors]
            axis.bar(positions, scores,
                     width=bar_width * 0.9, color=color_map[label], alpha=0.75, yerr=errors, capsize=3)

            for benchmark_index, benchmark in enumerate(BENCHMARKS):
                tasks = TASKS[benchmark]
                for task_index, task in enumerate(tasks):
                    task_score, _ = parse_score(row.get(task))
                    if task_score is None:
                        continue
                    task_offset = (task_index - (len(tasks) - 1) / 2) * bar_width * 0.25
                    axis.scatter(positions[benchmark_index] + task_offset, task_score,
                                 marker=marker_map[task], s=45, color=color_map[label], edgecolors="black", linewidths=0.6, zorder=3)

        axis.set_title(model)
        axis.set_xticks(range(len(BENCHMARKS)), BENCHMARKS)
        axis.set_xlabel("Benchmark")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Score (%)")
    # Create a combined legend for labels and tasks
    label_handles = [Patch(facecolor=color_map[label], alpha=0.75, label=label) for label in labels]
    task_handles = [
        Line2D([0], [0],
               marker=marker_map[task], color="none", 
               markerfacecolor="gray", markeredgecolor="black", label=task, markersize=7)
        for benchmark in BENCHMARKS
        for task in TASKS[benchmark]
    ]
    # Place the legend above the subplots, centered, with a maximum of 4 columns
    figure.legend(
        handles=label_handles + task_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1),
        ncol=min(4, len(label_handles + task_handles)),
    )
    # Adjust layout to make room for the legend
    figure.tight_layout(rect=(0, 0, 1, 0.9))

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = get_args()

    if len(args.input) != len(args.labels):
        raise ValueError("The number of input files must match the number of labels.")

    plot_results(args.input, args.labels, args.output)


if __name__ == "__main__":
    main()
