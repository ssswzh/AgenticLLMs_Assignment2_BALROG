import argparse
import csv
import tempfile
from math import sqrt
from pathlib import Path

from compare_summary import parse_score, plot_results
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
    for runs in runs_by_model.values():
        runs.sort(reverse=True, key=lambda item: item[0])
        for _, row in runs[:keep] if keep else runs:
            groups.setdefault((row["Temperature"], row["Model"]), []).append(row)

    rows = []
    for (temperature, model), group in groups.items():
        row = {"Temperature": temperature, "Model": model}
        for column in SCORE_COLUMNS:
            row[column] = average_cells([run.get(column, "") for run in group])
        rows.append(row)
    return sorted(rows, key=lambda row: (row["Temperature"], row["Model"]))


def write_table(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def plot_by_temperature(rows, output):
    temperatures = sorted({row["Temperature"] for row in rows})
    with tempfile.TemporaryDirectory() as directory:
        input_files = []
        for index, temperature in enumerate(temperatures):
            path = Path(directory) / f"temperature_{index}.tsv"
            write_table([row for row in rows if row["Temperature"] == temperature], path)
            input_files.append(path)
        plot_results(input_files, [f"temperature={value:g}" for value in temperatures], output)


def main():
    args = get_args()
    rows = summarize(args.dir, args.type, args.models, args.keep)
    if not rows:
        raise SystemExit(f"No {args.type!r} results found in {args.dir}")

    output = write_table(rows, args.output)
    plot = Path(args.plot) if args.plot else output.with_suffix(".png")
    plot_by_temperature(rows, plot)
    print(f"Saved summary to {output}")
    print(f"Saved plot to {plot}")


if __name__ == "__main__":
    main()
