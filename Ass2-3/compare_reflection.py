import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_LABELS = ["with reflection", "without reflection"]
DISPLAY_NAMES = {
    "babyai": "BabyAI",
    "textworld": "TextWorld",
    "babaisai": "BabaIsAI",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, nargs=2, help="With-reflection and without-reflection summary JSON files, in that order.")
    parser.add_argument("-l", "--labels", type=str, nargs=2, default=DEFAULT_LABELS, help="Labels for the two input files.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file for the plot.")
    return parser.parse_args()


def load_results(input_files):
    results = []
    for input_file in input_files:
        with Path(input_file).open("r", encoding="utf-8") as file:
            summary = json.load(file)
        environments = summary.get("environments")
        if not environments:
            raise ValueError(f"No environments found in {input_file}")
        results.append(environments)

    return results


def task_label(task):
    """Use the final path component as the subtype shown in the legend."""
    return task.rsplit("/", 1)[-1]


def _output_base(output_file):
    """Return the plot path without its optional file extension."""
    output_path = Path(output_file)
    return output_path.with_suffix("") if output_path.suffix else output_path


def _format_task_result(task_data):
    """Format progression, standard error, and average steps in one TSV cell."""
    if task_data is None:
        return ""
    return (
        f"{task_data['progression_percentage']:.2f}"
        f"±{task_data.get('standard_error', 0.0):.2f} "
        f"({task_data.get('average_steps', 0.0):.2f})"
    )


def _combine_trial_results(trial_results):
    """Combine environment summaries using their episode counts.

    The stored standard errors use population variance, so the within-group
    squared deviations can be recovered as ``(standard_error * n) ** 2``.
    """
    groups = [
        trial
        for trial in trial_results
        if trial is not None and trial.get("episodes_played", 0) > 0
    ]
    total_episodes = sum(group["episodes_played"] for group in groups)
    if total_episodes == 0:
        return None

    mean = sum(
        group["progression_percentage"] * group["episodes_played"]
        for group in groups
    ) / total_episodes
    squared_deviations = sum(
        (group.get("standard_error", 0.0) * group["episodes_played"]) ** 2
        + group["episodes_played"]
        * (group["progression_percentage"] - mean) ** 2
        for group in groups
    )
    standard_error = math.sqrt(squared_deviations) / total_episodes
    return mean, standard_error


def write_tsv_results(results, labels, output_file):
    """Write task/trial details and pooled trial averages to two TSV files."""
    output_base = _output_base(output_file)
    raw_output = output_base.with_name(output_base.name + "_task_trial.tsv")
    average_output = output_base.with_name(output_base.name + "_trial_average.tsv")
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    environment_names = []
    all_trial_indices = set()
    for environments in results:
        for environment, environment_data in environments.items():
            if environment not in environment_names:
                environment_names.append(environment)
            all_trial_indices.update(
                int(trial_idx) for trial_idx in environment_data.get("trials", {})
            )

    # One row per benchmark/task/trial. Each label cell uses
    # "progression±standard_error (average_steps)".
    with raw_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["Benchmark", "Task", "Trial", *labels])
        for environment in environment_names:
            task_names = []
            environment_trials = set()
            for environments in results:
                environment_data = environments.get(environment, {})
                trials = environment_data.get("trials", {})
                environment_trials.update(int(trial_idx) for trial_idx in trials)
                for trial in trials.values():
                    for task in trial.get("tasks", {}):
                        if task not in task_names:
                            task_names.append(task)

            for task in task_names:
                for trial_idx in sorted(environment_trials):
                    values = []
                    row_exists = False
                    for environments in results:
                        task_data = (
                            environments.get(environment, {})
                            .get("trials", {})
                            .get(str(trial_idx), {})
                            .get("tasks", {})
                            .get(task)
                        )
                        row_exists = row_exists or task_data is not None
                        values.append(_format_task_result(task_data))
                    if row_exists:
                        writer.writerow(
                            [
                                DISPLAY_NAMES.get(environment, environment),
                                task_label(task),
                                trial_idx,
                                *values,
                            ]
                        )

    # Pool all available environment/task episodes for each trial. The two
    # label columns contain "average progression±standard error".
    with average_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["Trial", *labels])
        for trial_idx in sorted(all_trial_indices):
            values = []
            for environments in results:
                trial_results = [
                    environment_data.get("trials", {}).get(str(trial_idx))
                    for environment_data in environments.values()
                ]
                combined = _combine_trial_results(trial_results)
                values.append(
                    "" if combined is None else f"{combined[0]:.2f}±{combined[1]:.2f}"
                )
            writer.writerow([trial_idx, *values])

    return raw_output, average_output


def plot_results(input_files, labels, output_file):
    results = load_results(input_files)

    # Keep the environment order from the JSON files.
    environment_names = []
    for environments in results:
        for environment in environments:
            if environment not in environment_names:
                environment_names.append(environment)

    color_map = {
        label: plt.get_cmap("tab10")(index % 10)
        for index, label in enumerate(labels)
    }
    task_names = []
    for environments in results:
        for environment, environment_data in environments.items():
            for trial in environment_data.get("trials", {}).values():
                for task in trial.get("tasks", {}):
                    task_key = (environment, task)
                    if task_key not in task_names:
                        task_names.append(task_key)

    task_markers = ["o", "s", "^", "D", "P", "X"]
    marker_map = {
        task_key: task_markers[index % len(task_markers)]
        for index, task_key in enumerate(task_names)
    }

    figure, axes = plt.subplots(
        1,
        len(environment_names),
        figsize=(max(8, 3 * len(environment_names)), 5),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    # Each subplot compares all trials for one environment.
    for axis, environment in zip(axes, environment_names):
        all_trial_indices = set()
        for label_index, (label, environments) in enumerate(zip(labels, results)):
            if environment not in environments:
                raise ValueError(f"Environment {environment!r} is missing from the {label!r} results")

            trials = environments[environment].get("trials", {})
            trial_indices = sorted(int(trial_idx) for trial_idx in trials)
            all_trial_indices.update(trial_indices)
            offset = (label_index - (len(labels) - 1) / 2) * 0.12
            positions = [trial_idx + offset for trial_idx in trial_indices]
            scores = [trials[str(trial_idx)]["progression_percentage"] for trial_idx in trial_indices]
            errors = [trials[str(trial_idx)].get("standard_error", 0.0) for trial_idx in trial_indices]

            axis.errorbar(
                positions,
                scores,
                yerr=errors,
                color=color_map[label],
                marker=".",
                linewidth=1.8,
                capsize=3,
                label=label,
                zorder=2,
            )

            # Task/subtype points use the same condition color and a distinct marker.
            for trial_idx, position in zip(trial_indices, positions):
                tasks = trials[str(trial_idx)].get("tasks", {})
                task_count = len(tasks)
                for task_index, (task, task_data) in enumerate(tasks.items()):
                    task_offset = (task_index - (task_count - 1) / 2) * 0.035
                    axis.scatter(
                        position + task_offset,
                        task_data["progression_percentage"],
                        marker=marker_map[(environment, task)],
                        s=45,
                        color=color_map[label],
                        edgecolors="black",
                        linewidths=0.6,
                        zorder=3,
                    )

        axis.set_title(DISPLAY_NAMES.get(environment, environment))
        axis.set_xticks(sorted(all_trial_indices))
        axis.set_xlabel("Trial")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Average Progress (%)")
    # Color identifies reflection condition; marker identifies task subtype.
    label_handles = [
        Line2D([0], [0], color=color_map[label], marker=".", linewidth=1.8, label=label)
        for label in labels
    ]
    task_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[task_key],
            color="none",
            markerfacecolor="gray",
            markeredgecolor="black",
            label=f"{DISPLAY_NAMES.get(task_key[0], task_key[0])}: {task_label(task_key[1])}",
            markersize=7,
        )
        for task_key in task_names
    ]
    figure.legend(
        handles=label_handles + task_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1),
        ncol=min(4, len(label_handles + task_handles)),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    args = get_args()
    results = load_results(args.input)
    plot_results(args.input, args.labels, args.output)
    raw_output, average_output = write_tsv_results(
        results, args.labels, args.output
    )
    print(f"Task/trial results written to {raw_output}")
    print(f"Trial averages written to {average_output}")


if __name__ == "__main__":
    main()
