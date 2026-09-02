"""Comprehensive multi-dimensional analysis of all completed TopoBox-3D runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("MGN-lite", "rigno", "Transolver", "GNOT", "GAOT", "TNO")
PROTOCOLS = ("A", "B", "C", "D")
DEGREES = (0, 1, 2)
SEEDS = (0, 1, 2)
CONFIGS = ("non_harmonic", "weak_harmonic", "balanced", "strong_harmonic")
POSITIVE_HARMONIC_CONFIGS = ("weak_harmonic", "balanced", "strong_harmonic")
METRICS = ("relative_l2", "harmonic_relative", "nonharmonic_relative")
VALID_HARMONIC_TASKS = {
    ("A", 1),
    ("A", 2),
    ("B", 1),
    ("C", 2),
    ("D", 1),
    ("D", 2),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ratio(
    iid: np.ndarray,
    ood: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    iid_idx = rng.integers(0, len(iid), size=(repeats, len(iid)))
    ood_idx = rng.integers(0, len(ood), size=(repeats, len(ood)))
    draws = ood[ood_idx].mean(axis=1) / (iid[iid_idx].mean(axis=1) + 1e-12)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def geometric_mean(values: pd.Series) -> float:
    numeric = values.astype(float).to_numpy()
    return float(np.exp(np.mean(np.log(numeric))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/topobox3d"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/comprehensive_analysis")
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)

    run_rows: list[dict] = []
    sample_rows: list[dict] = []
    quality_issues: list[str] = []

    for model in MODELS:
        for protocol in PROTOCOLS:
            for degree in DEGREES:
                for seed in SEEDS:
                    run_dir = (
                        args.root
                        / model
                        / f"protocol_{protocol}"
                        / f"k{degree}"
                        / f"seed_{seed}"
                    )
                    summary_path = run_dir / "summary.json"
                    if not summary_path.exists():
                        quality_issues.append(f"missing summary: {run_dir}")
                        continue
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if int(summary.get("completed_epoch", -1)) != 300:
                        quality_issues.append(
                            f"incomplete epoch: {run_dir} "
                            f"({summary.get('completed_epoch')})"
                        )
                    if (
                        degree == 2
                        and summary.get("cochain_normalization")
                        != "k2_per_sample_input_rms"
                    ):
                        quality_issues.append(f"invalid k2 normalization: {run_dir}")

                    row = {
                        "model": model,
                        "model_key": summary["model"],
                        "protocol": protocol,
                        "degree": degree,
                        "seed": seed,
                        "parameter_count": summary["parameter_count"],
                        "completed_epoch": summary["completed_epoch"],
                        "best_epoch": summary.get("best_epoch"),
                        "best_validation_relative_mse": summary.get(
                            "best_validation_relative_mse"
                        ),
                    }
                    for split in ("validation", "test_iid", "test_ood"):
                        split_data = summary["splits"][split]
                        row[f"{split}_samples"] = split_data["samples"]
                        row[f"{split}_geometry_count"] = split_data[
                            "geometry_clustered_relative_l2"
                        ]["geometry_count"]
                        for metric in METRICS:
                            row[f"{split}_{metric}_mean"] = split_data[metric]["mean"]
                            row[f"{split}_{metric}_median"] = split_data[metric][
                                "median"
                            ]
                    row["relative_l2_ratio"] = (
                        row["test_ood_relative_l2_mean"]
                        / row["test_iid_relative_l2_mean"]
                    )
                    row["nonharmonic_ratio"] = (
                        row["test_ood_nonharmonic_relative_mean"]
                        / row["test_iid_nonharmonic_relative_mean"]
                    )
                    if (protocol, degree) in VALID_HARMONIC_TASKS:
                        row["harmonic_ratio_summary_raw"] = (
                            row["test_ood_harmonic_relative_mean"]
                            / row["test_iid_harmonic_relative_mean"]
                        )
                    else:
                        row["harmonic_ratio_summary_raw"] = None
                    run_rows.append(row)

                    for split in ("test_iid", "test_ood"):
                        path = run_dir / f"{split}.jsonl"
                        if not path.exists():
                            quality_issues.append(f"missing samples: {path}")
                            continue
                        with path.open(encoding="utf-8") as handle:
                            for line in handle:
                                if not line.strip():
                                    continue
                                item = json.loads(line)
                                sample_rows.append(
                                    {
                                        "model": model,
                                        "protocol": protocol,
                                        "degree": degree,
                                        "seed": seed,
                                        "split": split,
                                        "geometry_id": item["geometry_id"],
                                        "config": item["config_name"],
                                        "beta1": item["beta1"],
                                        "beta2": item["beta2"],
                                        "rho_h": float(
                                            item["realized_energy_fractions"][2]
                                        ),
                                        "relative_l2": float(item["relative_l2"]),
                                        "relative_mse": float(item["relative_mse"]),
                                        "harmonic_relative": float(
                                            item["harmonic_relative"]
                                        ),
                                        "nonharmonic_relative": float(
                                            item["nonharmonic_relative"]
                                        ),
                                    }
                                )

    samples = pd.DataFrame(sample_rows)
    runs = pd.DataFrame(run_rows)
    group_rows: list[dict] = []

    for model in MODELS:
        for protocol in PROTOCOLS:
            for degree in DEGREES:
                task = samples[
                    (samples["model"] == model)
                    & (samples["protocol"] == protocol)
                    & (samples["degree"] == degree)
                ]
                metric_configs: list[tuple[str, str, tuple[str, ...]]] = []
                for metric in ("relative_l2", "nonharmonic_relative"):
                    metric_configs.append((metric, "all", CONFIGS))
                    metric_configs.extend(
                        (metric, config, (config,)) for config in CONFIGS
                    )
                if (protocol, degree) in VALID_HARMONIC_TASKS:
                    metric_configs.append(
                        (
                            "harmonic_relative",
                            "positive_harmonic",
                            POSITIVE_HARMONIC_CONFIGS,
                        )
                    )
                    metric_configs.extend(
                        ("harmonic_relative", config, (config,))
                        for config in POSITIVE_HARMONIC_CONFIGS
                    )

                for metric, config_label, selected_configs in metric_configs:
                    filtered = task[task["config"].isin(selected_configs)]
                    geometry = (
                        filtered.groupby(["split", "geometry_id"], as_index=False)[
                            metric
                        ]
                        .mean()
                        .rename(columns={metric: "value"})
                    )
                    iid = geometry[geometry["split"] == "test_iid"]["value"].to_numpy()
                    ood = geometry[geometry["split"] == "test_ood"]["value"].to_numpy()
                    ratio = float(np.mean(ood) / (np.mean(iid) + 1e-12))
                    low, high = bootstrap_ratio(
                        iid, ood, args.bootstrap_repeats, rng
                    )
                    seed_ratios = []
                    for seed in SEEDS:
                        seed_data = filtered[filtered["seed"] == seed]
                        seed_means = seed_data.groupby("split")[metric].mean()
                        seed_ratios.append(
                            float(
                                seed_means["test_ood"]
                                / (seed_means["test_iid"] + 1e-12)
                            )
                        )
                    group_rows.append(
                        {
                            "model": model,
                            "protocol": protocol,
                            "degree": degree,
                            "metric": metric,
                            "config": config_label,
                            "iid_mean": float(np.mean(iid)),
                            "iid_median": float(np.median(iid)),
                            "iid_geometry_sd": float(np.std(iid, ddof=1)),
                            "ood_mean": float(np.mean(ood)),
                            "ood_median": float(np.median(ood)),
                            "ood_geometry_sd": float(np.std(ood, ddof=1)),
                            "ood_minus_iid": float(np.mean(ood) - np.mean(iid)),
                            "degradation_ratio": ratio,
                            "ratio_ci95_low": low,
                            "ratio_ci95_high": high,
                            "seed_ratio_mean": float(np.mean(seed_ratios)),
                            "seed_ratio_sd": float(np.std(seed_ratios, ddof=1)),
                            "iid_geometry_count": len(iid),
                            "ood_geometry_count": len(ood),
                            "significant_degradation": low > 1,
                            "significant_improvement": high < 1,
                        }
                    )

    groups = pd.DataFrame(group_rows)
    primary = groups[
        ((groups["metric"] == "relative_l2") & (groups["config"] == "all"))
        | (
            (groups["metric"] == "nonharmonic_relative")
            & (groups["config"] == "all")
        )
        | (
            (groups["metric"] == "harmonic_relative")
            & (groups["config"] == "positive_harmonic")
        )
    ].copy()
    primary["ood_rank"] = primary.groupby(
        ["protocol", "degree", "metric"]
    )["ood_mean"].rank(method="min", ascending=True)
    primary["iid_rank"] = primary.groupby(
        ["protocol", "degree", "metric"]
    )["iid_mean"].rank(method="min", ascending=True)
    primary["robustness_rank"] = primary.groupby(
        ["protocol", "degree", "metric"]
    )["degradation_ratio"].rank(method="min", ascending=True)

    overall = primary[primary["metric"] == "relative_l2"].copy()
    model_summary = (
        overall.groupby("model")
        .agg(
            mean_iid_relative_l2=("iid_mean", "mean"),
            mean_ood_relative_l2=("ood_mean", "mean"),
            geometric_mean_ratio=("degradation_ratio", geometric_mean),
            median_ratio=("degradation_ratio", "median"),
            mean_iid_rank=("iid_rank", "mean"),
            mean_ood_rank=("ood_rank", "mean"),
            mean_robustness_rank=("robustness_rank", "mean"),
            significant_degradation_tasks=("significant_degradation", "sum"),
            significant_improvement_tasks=("significant_improvement", "sum"),
        )
        .reset_index()
    )
    model_summary["absolute_ood_wins"] = model_summary["model"].map(
        (overall[overall["ood_rank"] == 1].groupby("model").size()).to_dict()
    ).fillna(0)
    model_summary["robustness_wins"] = model_summary["model"].map(
        (overall[overall["robustness_rank"] == 1].groupby("model").size()).to_dict()
    ).fillna(0)
    model_summary = model_summary.sort_values("mean_ood_rank")

    protocol_summary = (
        overall.groupby("protocol")
        .agg(
            mean_iid_relative_l2=("iid_mean", "mean"),
            mean_ood_relative_l2=("ood_mean", "mean"),
            geometric_mean_ratio=("degradation_ratio", geometric_mean),
            median_ratio=("degradation_ratio", "median"),
            min_ratio=("degradation_ratio", "min"),
            max_ratio=("degradation_ratio", "max"),
            significant_degradation_cells=("significant_degradation", "sum"),
            significant_improvement_cells=("significant_improvement", "sum"),
        )
        .reset_index()
    )
    equation_summary = (
        overall.groupby("degree")
        .agg(
            mean_iid_relative_l2=("iid_mean", "mean"),
            mean_ood_relative_l2=("ood_mean", "mean"),
            geometric_mean_ratio=("degradation_ratio", geometric_mean),
            median_ratio=("degradation_ratio", "median"),
            min_ratio=("degradation_ratio", "min"),
            max_ratio=("degradation_ratio", "max"),
            significant_degradation_cells=("significant_degradation", "sum"),
            significant_improvement_cells=("significant_improvement", "sum"),
        )
        .reset_index()
    )
    dataset_equation_summary = (
        overall.groupby(["protocol", "degree"])
        .agg(
            mean_iid_relative_l2=("iid_mean", "mean"),
            mean_ood_relative_l2=("ood_mean", "mean"),
            geometric_mean_ratio=("degradation_ratio", geometric_mean),
            median_ratio=("degradation_ratio", "median"),
            significant_degradation_models=("significant_degradation", "sum"),
            significant_improvement_models=("significant_improvement", "sum"),
        )
        .reset_index()
    )

    protocol_vs_a_rows = []
    ratio_pivot = overall.pivot_table(
        index=["model", "degree"],
        columns="protocol",
        values="degradation_ratio",
    )
    for protocol in ("B", "C", "D"):
        paired_log = np.log(
            ratio_pivot[protocol].to_numpy() / ratio_pivot["A"].to_numpy()
        )
        draw_indices = rng.integers(
            0,
            len(paired_log),
            size=(10000, len(paired_log)),
        )
        draws = paired_log[draw_indices].mean(axis=1)
        low, high = np.quantile(draws, (0.025, 0.975))
        positive_count = int(np.sum(paired_log > 0))
        sign_p = sum(
            math.comb(len(paired_log), value)
            for value in range(positive_count, len(paired_log) + 1)
        ) / (2 ** len(paired_log))
        protocol_vs_a_rows.append(
            {
                "protocol": protocol,
                "paired_cells": len(paired_log),
                "cells_harder_than_A": positive_count,
                "ratio_of_geometric_mean_degradation_vs_A": float(
                    np.exp(np.mean(paired_log))
                ),
                "bootstrap_ci95_low": float(np.exp(low)),
                "bootstrap_ci95_high": float(np.exp(high)),
                "one_sided_sign_test_p": float(sign_p),
            }
        )

    component_rows = []
    for model in MODELS:
        for protocol, degree in sorted(VALID_HARMONIC_TASKS):
            harmonic = primary[
                (primary["model"] == model)
                & (primary["protocol"] == protocol)
                & (primary["degree"] == degree)
                & (primary["metric"] == "harmonic_relative")
            ].iloc[0]
            nonharmonic = primary[
                (primary["model"] == model)
                & (primary["protocol"] == protocol)
                & (primary["degree"] == degree)
                & (primary["metric"] == "nonharmonic_relative")
            ].iloc[0]
            component_rows.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "degree": degree,
                    "harmonic_iid": harmonic["iid_mean"],
                    "harmonic_ood": harmonic["ood_mean"],
                    "harmonic_ratio": harmonic["degradation_ratio"],
                    "nonharmonic_iid": nonharmonic["iid_mean"],
                    "nonharmonic_ood": nonharmonic["ood_mean"],
                    "nonharmonic_ratio": nonharmonic["degradation_ratio"],
                    "harmonic_ratio_over_nonharmonic_ratio": (
                        harmonic["degradation_ratio"]
                        / nonharmonic["degradation_ratio"]
                    ),
                }
            )

    config_summary = (
        groups[
            (groups["metric"] == "relative_l2")
            & (groups["config"].isin(CONFIGS))
        ]
        .groupby(["protocol", "degree", "config"])
        .agg(
            mean_iid_relative_l2=("iid_mean", "mean"),
            mean_ood_relative_l2=("ood_mean", "mean"),
            geometric_mean_ratio=("degradation_ratio", geometric_mean),
        )
        .reset_index()
    )

    write_csv(args.output / "run_metrics_complete.csv", run_rows)
    write_csv(args.output / "group_statistics_complete.csv", group_rows)
    write_csv(
        args.output / "rankings.csv",
        primary.replace({np.nan: None}).to_dict("records"),
    )
    write_csv(
        args.output / "model_comparison.csv",
        model_summary.replace({np.nan: None}).to_dict("records"),
    )
    write_csv(
        args.output / "dataset_comparison.csv",
        protocol_summary.replace({np.nan: None}).to_dict("records"),
    )
    write_csv(
        args.output / "equation_comparison.csv",
        equation_summary.replace({np.nan: None}).to_dict("records"),
    )
    write_csv(
        args.output / "dataset_equation_comparison.csv",
        dataset_equation_summary.replace({np.nan: None}).to_dict("records"),
    )
    write_csv(args.output / "protocol_vs_A.csv", protocol_vs_a_rows)
    write_csv(args.output / "error_component_comparison.csv", component_rows)
    write_csv(
        args.output / "initial_condition_comparison.csv",
        config_summary.replace({np.nan: None}).to_dict("records"),
    )

    headline = {
        "expected_runs": 216,
        "valid_runs": len(run_rows),
        "sample_records": len(sample_rows),
        "quality_issues": quality_issues,
        "best_absolute_model_by_mean_ood_rank": model_summary.iloc[0]["model"],
        "best_robustness_model_by_mean_rank": model_summary.sort_values(
            "mean_robustness_rank"
        ).iloc[0]["model"],
        "hardest_protocol_by_mean_ood_error": protocol_summary.sort_values(
            "mean_ood_relative_l2", ascending=False
        ).iloc[0]["protocol"],
        "largest_protocol_degradation": protocol_summary.sort_values(
            "geometric_mean_ratio", ascending=False
        ).iloc[0]["protocol"],
        "hardest_equation_by_mean_ood_error": int(
            equation_summary.sort_values(
                "mean_ood_relative_l2", ascending=False
            ).iloc[0]["degree"]
        ),
        "largest_equation_degradation": int(
            equation_summary.sort_values(
                "geometric_mean_ratio", ascending=False
            ).iloc[0]["degree"]
        ),
        "group_statistics_rows": len(group_rows),
    }
    (args.output / "headline.json").write_text(
        json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(headline, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
