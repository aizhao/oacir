#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


DOMAINS = ("car", "fashion", "product", "landmark")
REPORT_METRICS = ("R_ID@1", "R@1", "R@5", "arithmetic_mean", "primary3")
RAW_METRICS = ("R_ID@1", "R_ID@3", "R_ID@5", "R@1", "R@5", "R@10", "R@50")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--latent-dir", required=True, type=Path)
    parser.add_argument("--box-dir", required=True, type=Path)
    parser.add_argument("--identity-dir", required=True, type=Path)
    parser.add_argument("--latent-epoch", default=42, type=int)
    parser.add_argument("--box-epoch", default=46, type=int)
    return parser.parse_args()


def derive(row):
    values = {key: float(row[key]) for key in RAW_METRICS}
    values["arithmetic_mean"] = sum(
        values[key] for key in ("R@1", "R@5", "R@10", "R@50")
    ) / 4.0
    values["primary3"] = sum(
        values[key] for key in ("R_ID@1", "R@1", "R@5")
    ) / 3.0
    return values


def load_epoch(path, epoch):
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if int(row["epoch"]) == epoch:
                return derive(row)
    raise ValueError(f"Epoch {epoch} not found in {path}")


def load_config(path, fusion, lambda_id):
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if (
                abs(float(row["fusion"]) - fusion) < 1e-9
                and abs(float(row["lambda_id"]) - lambda_id) < 1e-9
            ):
                return derive(row)
    raise ValueError(f"Config ({fusion}, {lambda_id}) not found in {path}")


def aggregate_domains(domains):
    result = {
        metric: sum(domains[domain][metric] for domain in DOMAINS) / len(DOMAINS)
        for metric in REPORT_METRICS
    }
    result["domains"] = domains
    return result


def main():
    args = parse_args()
    experiments = {
        "A0_latent": aggregate_domains(
            {
                domain: load_epoch(
                    args.latent_dir / f"{domain}_validation_metrics.csv",
                    args.latent_epoch,
                )
                for domain in DOMAINS
            }
        ),
        "A1_box_query": aggregate_domains(
            {
                domain: load_epoch(
                    args.box_dir / f"{domain}_validation_metrics.csv",
                    args.box_epoch,
                )
                for domain in DOMAINS
            }
        ),
        "A2_zero_shot_hybrid": aggregate_domains(
            {
                domain: load_config(
                    args.identity_dir
                    / f"ablation_a2_zero_shot_{domain}"
                    / f"{domain}_score_search_results.csv",
                    0.4,
                    1.0,
                )
                for domain in DOMAINS
            }
        ),
        "B2_latent_discovery_identity": aggregate_domains(
            {
                domain: load_config(
                    args.identity_dir
                    / f"ablation_b2_latent_discovery_{domain}"
                    / f"{domain}_score_search_results.csv",
                    0.4,
                    1.0,
                )
                for domain in DOMAINS
            }
        ),
        "A3_finetuned_lambda1": aggregate_domains(
            {
                domain: load_config(
                    args.identity_dir
                    / f"score_search_e2_pareto_{domain}"
                    / f"{domain}_score_search_results.csv",
                    0.4,
                    1.0,
                )
                for domain in DOMAINS
            }
        ),
        "A4_finetuned_lambda1_25": aggregate_domains(
            {
                domain: load_config(
                    args.identity_dir
                    / f"score_search_e2_pareto_{domain}"
                    / f"{domain}_score_search_results.csv",
                    0.4,
                    1.25,
                )
                for domain in DOMAINS
            }
        ),
    }

    baseline = experiments["A0_latent"]
    previous = None
    rows = []
    for name, values in experiments.items():
        row = {"experiment": name}
        for metric in REPORT_METRICS:
            row[metric] = values[metric]
            row[f"delta_latent_{metric}"] = values[metric] - baseline[metric]
            row[f"delta_previous_{metric}"] = (
                0.0 if previous is None else values[metric] - previous[metric]
            )
        rows.append(row)
        previous = values

    fields = ["experiment"]
    for metric in REPORT_METRICS:
        fields.extend((metric, f"delta_latent_{metric}", f"delta_previous_{metric}"))
    csv_path = args.identity_dir / "ablation_a0_a4_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {"experiments": experiments, "summary_rows": rows}
    json_path = args.identity_dir / "ablation_a0_a4_summary.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
