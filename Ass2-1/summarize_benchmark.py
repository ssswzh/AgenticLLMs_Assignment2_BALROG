import argparse
import csv
import json
from math import sqrt
from pathlib import Path


MODELS = {
    "qwen3:4b-instruct": "Qwen3:4B-Instruct",
    "gpt-6.1-sol": "GPT-6.1-Sol",
}
BENCHMARK_NAMES = {
    "babyai": "BabyAI",
    "textworld": "TextWorld",
    "babaisai": "BabaIsAI",
}
# Map task names in each benchmark JSON file to the corresponding output columns
BENCHMARKS = {
    "babyai": {
        "BabyAI-MixedTrainLocal-v0/goto": "BabyAI-goto",
        "BabyAI-MixedTrainLocal-v0/putnext": "BabyAI-putnext",
    },
    "textworld": {
        "coin_collector": "TextWorld-coin_collector",
        "treasure_hunter": "TextWorld-treasure_hunter",
    },
    "babaisai": {
        "env/goto_win": "BabaIsAI-goto_win",
        "env/make_win": "BabaIsAI-make_win",
    },
}
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


def get_args():
    parser = argparse.ArgumentParser(description="Summarize benchmark results.")
    parser.add_argument("-d", "--dir", type=str, required=True, help="Directory prefix containing the benchmark results.")
    parser.add_argument("-t", "--type", type=str, default="naive", choices=["naive", "cot", "few_shot"], help="Type of summary to generate.")
    parser.add_argument("-o", "--output", type=str, default="summary.tsv", help="Output TSV file for the summary.")
    return parser.parse_args()


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def get_score(data, value_key="progression_percentage", include_average_steps=False):
    average_steps = data.get("average_steps") if include_average_steps else None
    return data.get(value_key), data.get("standard_error"), average_steps


def format_score(score):
    value, standard_error, average_steps = score
    if value is None:
        return ""
    if standard_error is None:
        result = str(round(value, 2))
    else:
        result = f"{round(value, 2)}±{round(standard_error, 2)}"
    if average_steps is not None:
        result += f" ({round(average_steps, 2)})"
    return result


def summarize_benchmark(input_dir, summary_type="naive"):
    input_dir = Path(input_dir)
    # For example, results/exercise1 also matches results/exercise1_textworld
    result_dirs = [path for path in input_dir.parent.glob(f"{input_dir.name}*") if path.is_dir()]
    if not result_dirs:
        raise ValueError(f"No results directories match: {input_dir}*")

    results = {}
    # Process older runs first so newer results overwrite repeated benchmarks
    summary_files = [file for directory in result_dirs for file in directory.rglob("summary.json")]
    summary_files.sort(key=lambda path: path.stat().st_mtime_ns)
    for summary_file in summary_files:
        summary_data = load_json(summary_file)
        if summary_data.get("agent", {}).get("type") != summary_type:
            continue

        model_id = summary_data.get("client", {}).get("model_id")
        if not model_id:
            raise ValueError(f"Missing client.model_id in {summary_file}")

        row = results.setdefault(model_id, {"Model": MODELS.get(model_id, model_id)})
        # Return the average progress and standard error for the overall benchmark
        row["Average Progress"] = get_score(summary_data, "average_progress")

        # Benchmark summaries contain the environment and individual task scores
        for benchmark_file in summary_file.parent.rglob("*_summary.json"):
            benchmark_name = benchmark_file.name.removesuffix("_summary.json")
            if benchmark_name not in BENCHMARK_NAMES:
                continue

            benchmark_data = load_json(benchmark_file)
            # For benchmark summaries, return average progress, standard error, and average steps
            row[BENCHMARK_NAMES[benchmark_name]] = get_score(benchmark_data, include_average_steps=True)
            # For each task in the benchmark, return the average progress and standard error for that task
            for task_name, column_name in BENCHMARKS[benchmark_name].items():
                if task_name in benchmark_data.get("tasks", {}):
                    row[column_name] = get_score(benchmark_data["tasks"][task_name])

    # Recalculate the overall score after combining separately executed benchmarks
    benchmark_columns = list(BENCHMARK_NAMES.values())
    for row in results.values():
        benchmark_scores = [row[column] for column in benchmark_columns if column in row]
        if benchmark_scores:
            values = [score[0] for score in benchmark_scores]
            errors = [score[1] for score in benchmark_scores]
            average = sum(values) / len(values)
            standard_error = sqrt(sum(error ** 2 for error in errors)) / len(errors)
            row["Average Progress"] = average, standard_error, None
        # Format output: avg_progress +- standard_error (average_steps)
        for column in COLUMNS[1:]:
            if column in row:
                row[column] = format_score(row[column])

    model_order = [model_id for model_id in MODELS if model_id in results]
    model_order.extend(model_id for model_id in results if model_id not in MODELS)
    return [results[model_id] for model_id in model_order]


def write_table(rows, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_file


def print_table(rows):
    print("\t".join(COLUMNS))
    for row in rows:
        print("\t".join(str(row.get(column, "")) for column in COLUMNS))


def main():
    args = get_args()
    rows = summarize_benchmark(args.dir, args.type)
    if not rows:
        raise SystemExit(f"No {args.type!r} summary.json files found in {args.dir}")

    output_file = write_table(rows, args.output)
    print_table(rows)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
