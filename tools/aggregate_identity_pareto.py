#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DOMAINS = ("car", "fashion", "product", "landmark")
METRICS = (
    "R_ID@1",
    "R_ID@3",
    "R_ID@5",
    "R@1",
    "R@5",
    "R@10",
    "R@50",
    "Avg7",
    "primary3",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--baseline-epoch", default=46, type=int)
    parser.add_argument("--r1-tolerance", default=0.2, type=float)
    parser.add_argument("--default-fusion", default=0.4, type=float)
    parser.add_argument("--default-lambda", default=1.25, type=float)
    return parser.parse_args()


def load_search(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return {
            (float(row["fusion"]), float(row["lambda_id"])): {
                metric: float(row[metric]) for metric in METRICS
            }
            for row in rows
        }


def load_baseline(path, epoch):
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if int(row["epoch"]) == epoch:
                result = {
                    metric: float(row[metric])
                    for metric in METRICS
                    if metric in row
                }
                result["arithmetic_mean"] = float(row["arithmetic_mean"])
                result["primary3"] = (
                    result["R_ID@1"] + result["R@1"] + result["R@5"]
                ) / 3.0
                return result
    raise ValueError(f"Epoch {epoch} not found in {path}")


def compact(row):
    return {
        key: row[key]
        for key in (
            "fusion",
            "lambda_id",
            "R_ID@1",
            "R@1",
            "R@5",
            "arithmetic_mean",
            "primary3",
        )
    } | {"domains": row["domains"]}


def main():
    args = parse_args()
    per_domain = {
        domain: load_search(
            args.run_dir
            / f"score_search_e2_pareto_{domain}"
            / f"{domain}_score_search_results.csv"
        )
        for domain in DOMAINS
    }
    common = set.intersection(*(set(per_domain[domain]) for domain in DOMAINS))

    aggregate = []
    for fusion, lambda_id in common:
        row = {"fusion": fusion, "lambda_id": lambda_id}
        for metric in METRICS:
            row[metric] = sum(
                per_domain[domain][(fusion, lambda_id)][metric]
                for domain in DOMAINS
            ) / len(DOMAINS)
        row["arithmetic_mean"] = sum(
            row[metric] for metric in ("R@1", "R@5", "R@10", "R@50")
        ) / 4.0
        row["domains"] = {
            domain: per_domain[domain][(fusion, lambda_id)] for domain in DOMAINS
        }
        aggregate.append(row)

    baseline_domains = {
        domain: load_baseline(
            args.baseline_dir / f"{domain}_validation_metrics.csv",
            args.baseline_epoch,
        )
        for domain in DOMAINS
    }
    baseline = {
        metric: sum(baseline_domains[domain][metric] for domain in DOMAINS)
        / len(DOMAINS)
        for metric in ("R_ID@1", "R@1", "R@5", "arithmetic_mean", "primary3")
    }
    default = next(
        row
        for row in aggregate
        if row["fusion"] == args.default_fusion
        and row["lambda_id"] == args.default_lambda
    )

    def best_with_threshold(threshold):
        feasible = [row for row in aggregate if row["R@1"] >= threshold]
        best = max(
            feasible,
            key=lambda row: (row["R_ID@1"], row["primary3"], row["R@1"]),
        )
        return threshold, len(feasible), best

    box_constraint = best_with_threshold(baseline["R@1"] - args.r1_tolerance)
    e2_constraint = best_with_threshold(default["R@1"] - args.r1_tolerance)
    unconstrained = max(
        aggregate,
        key=lambda row: (row["R_ID@1"], row["primary3"]),
    )
    beats_box = [row for row in aggregate if row["R_ID@1"] > baseline["R_ID@1"]]
    least_r1_cost_to_beat_box = max(beats_box, key=lambda row: row["R@1"])

    fields = [
        "fusion",
        "lambda_id",
        *METRICS,
        "arithmetic_mean",
    ]
    csv_path = args.run_dir / "score_search_e2_pareto_global.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            aggregate,
            key=lambda item: (item["R_ID@1"], item["primary3"]),
            reverse=True,
        ):
            writer.writerow({key: row[key] for key in fields})

    summary = {
        "config_count": len(aggregate),
        "r1_tolerance": args.r1_tolerance,
        "box_query_baseline": baseline,
        "e2_default": compact(default),
        "box_query_constraint": {
            "threshold": box_constraint[0],
            "feasible_count": box_constraint[1],
            "best": compact(box_constraint[2]),
        },
        "e2_default_constraint": {
            "threshold": e2_constraint[0],
            "feasible_count": e2_constraint[1],
            "best": compact(e2_constraint[2]),
        },
        "least_r1_cost_to_beat_box_rid1": compact(least_r1_cost_to_beat_box),
        "best_unconstrained_rid1": compact(unconstrained),
    }
    json_path = args.run_dir / "score_search_e2_pareto_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
