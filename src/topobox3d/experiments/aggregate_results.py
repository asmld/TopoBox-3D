"""Aggregate run summaries and geometry-clustered IID/OOD degradation ratios."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("relative_l2", "harmonic_relative", "nonharmonic_relative")
EXPECTED_K2_NORMALIZATION = "k2_per_sample_input_rms"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clustered_values(records: list[dict], metric: str, config: str | None):
    groups = defaultdict(list)
    for record in records:
        if config is not None and record["config_name"] != config:
            continue
        groups[record["geometry_id"]].append(float(record[metric]))
    return np.asarray([np.mean(items) for items in groups.values()])


def hierarchical_ratio_interval(
    seed_pairs: list[tuple[np.ndarray, np.ndarray]],
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    observed = np.mean(
        [np.mean(ood) / (np.mean(iid) + 1e-8) for iid, ood in seed_pairs]
    )
    draws = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(seed_pairs), size=len(seed_pairs))
        ratios = []
        for index in chosen:
            iid, ood = seed_pairs[index]
            iid_draw = rng.choice(iid, size=len(iid), replace=True)
            ood_draw = rng.choice(ood, size=len(ood), replace=True)
            ratios.append(np.mean(ood_draw) / (np.mean(iid_draw) + 1e-8))
        draws.append(np.mean(ratios))
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(observed), float(low), float(high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/topobox3d"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/aggregate_results")
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--required-epochs", type=int, default=300)
    args = parser.parse_args()
    summaries = []
    grouped_runs = defaultdict(list)
    excluded_runs = []
    for summary_path in args.root.rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        required = {"model", "protocol", "degree", "seed", "splits"}
        if not required.issubset(summary):
            continue
        completed_epoch = summary.get("completed_epoch")
        if completed_epoch is None:
            history_path = summary_path.parent / "history.json"
            if history_path.exists():
                history = json.loads(history_path.read_text(encoding="utf-8"))
                if history:
                    completed_epoch = int(history[-1]["epoch"])
        reason = None
        if completed_epoch is None or int(completed_epoch) < args.required_epochs:
            reason = (
                f"fixed training budget incomplete: "
                f"{completed_epoch}/{args.required_epochs}"
            )
        elif (
            int(summary["degree"]) == 2
            and summary.get("cochain_normalization")
            != EXPECTED_K2_NORMALIZATION
        ):
            reason = (
                "legacy k=2 checkpoint without "
                f"{EXPECTED_K2_NORMALIZATION}"
            )
        if reason is not None:
            excluded_runs.append(
                {"path": str(summary_path.parent), "reason": reason}
            )
            continue
        summaries.append(summary)
        grouped_runs[
            (summary["model"], summary["protocol"], int(summary["degree"]))
        ].append(summary_path.parent)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "excluded_runs.json").write_text(
        json.dumps(excluded_runs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output / "run_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "model",
            "protocol",
            "degree",
            "seed",
            "parameters",
            "best_epoch",
            "iid_relative_l2",
            "ood_relative_l2",
            "degradation_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            iid = item["splits"]["test_iid"]["relative_l2"]["mean"]
            ood = item["splits"]["test_ood"]["relative_l2"]["mean"]
            writer.writerow(
                {
                    "model": item["model"],
                    "protocol": item["protocol"],
                    "degree": item["degree"],
                    "seed": item["seed"],
                    "parameters": item["parameter_count"],
                    "best_epoch": item.get("best_epoch", ""),
                    "iid_relative_l2": iid,
                    "ood_relative_l2": ood,
                    "degradation_ratio": ood / (iid + 1e-8),
                }
            )

    group_rows = []
    configs: list[str | None] = [
        None,
        "non_harmonic",
        "weak_harmonic",
        "balanced",
        "strong_harmonic",
    ]
    for group_key, run_dirs in sorted(grouped_runs.items()):
        model, protocol, degree = group_key
        for metric in METRICS:
            # k=0 targets deliberately suppress the constant harmonic mode;
            # its relative harmonic denominator is effectively zero.
            if degree == 0 and metric == "harmonic_relative":
                continue
            for config in configs:
                seed_pairs = []
                for run_dir in run_dirs:
                    iid_path = run_dir / "test_iid.jsonl"
                    ood_path = run_dir / "test_ood.jsonl"
                    if not iid_path.exists() or not ood_path.exists():
                        continue
                    iid = clustered_values(read_jsonl(iid_path), metric, config)
                    ood = clustered_values(read_jsonl(ood_path), metric, config)
                    if len(iid) and len(ood):
                        seed_pairs.append((iid, ood))
                if not seed_pairs:
                    continue
                estimate, low, high = hierarchical_ratio_interval(
                    seed_pairs,
                    args.bootstrap_repeats,
                    args.seed + degree,
                )
                group_rows.append(
                    {
                        "model": model,
                        "protocol": protocol,
                        "degree": degree,
                        "metric": metric,
                        "config": config or "all",
                        "seed_count": len(seed_pairs),
                        "ratio": estimate,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    if group_rows:
        with (args.output / "group_degradation.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
            writer.writeheader()
            writer.writerows(group_rows)
    print(
        json.dumps(
            {
                "run_count": len(summaries),
                "excluded_run_count": len(excluded_runs),
                "group_rows": len(group_rows),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
