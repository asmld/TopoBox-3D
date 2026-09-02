"""Jensen spectral-broadening diagnostics for relative final-time error."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path.cwd()
OUT = ROOT / "results/03_error_source_univariate_regression"
SUPPORT = OUT / "supporting_data"
FIGURES = OUT / "figures/jensen_relative"
SOURCE = SUPPORT / "sample_level_analysis.csv"
MODELS = ["MGN-lite", "RIGNO", "Transolver", "GNOT", "GAOT", "TNO"]
TASKS = [f"{protocol}-k{degree}" for protocol in "ABCD" for degree in (0, 1, 2)]
KAPPA_T = 0.1


def ranked(values: pd.Series) -> np.ndarray:
    return stats.rankdata(values.to_numpy(dtype=float), method="average")


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return np.nan, np.nan
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def residualize(y: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), *controls])
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficient


def partial_spearman(
    x: pd.Series, y: pd.Series, controls: list[pd.Series]
) -> tuple[float, float]:
    rx = ranked(x)
    ry = ranked(y)
    control_ranks = [ranked(control) for control in controls]
    ex = residualize(rx, control_ranks)
    ey = residualize(ry, control_ranks)
    result = stats.pearsonr(ex, ey)
    return float(result.statistic), float(result.pvalue)


def rank_r2(y: pd.Series, predictors: list[pd.Series]) -> float:
    ry = ranked(y)
    design = np.column_stack(
        [np.ones(len(ry)), *[ranked(predictor) for predictor in predictors]]
    )
    coefficient, *_ = np.linalg.lstsq(design, ry, rcond=None)
    fitted = design @ coefficient
    total = np.sum((ry - ry.mean()) ** 2)
    return float(1.0 - np.sum((ry - fitted) ** 2) / total)


def scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return frame
    return frame[frame["split"] == f"test_{scope}"]


def cell_correlations(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, protocol, degree, task), cell in samples.groupby(
        ["model", "protocol", "degree", "task"], sort=False
    ):
        for scope in ("all", "iid", "ood"):
            selected = scope_frame(cell, scope)
            error = selected["E_final"]
            rayleigh = selected["initial_rayleigh_quotient"]
            retention = selected["target_norm_retention"]
            gap = selected["jensen_gap"]
            rho_j, p_j = safe_spearman(gap, error)
            rho_r, p_r = safe_spearman(rayleigh, error)
            rho_q, p_q = safe_spearman(retention, error)
            rho_rq, _ = safe_spearman(rayleigh, retention)
            partial_r, partial_r_p = partial_spearman(
                gap, error, [rayleigh]
            )
            partial_q, partial_q_p = partial_spearman(
                gap, error, [retention]
            )
            partial_rq, partial_rq_p = partial_spearman(
                gap, error, [rayleigh, retention]
            )
            r2_r = rank_r2(error, [rayleigh])
            r2_q = rank_r2(error, [retention])
            r2_rq = rank_r2(error, [rayleigh, retention])
            rows.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "degree": int(degree),
                    "task": task,
                    "scope": scope,
                    "n": len(selected),
                    "rho_J_error": rho_j,
                    "p_J_error": p_j,
                    "rho_R_error": rho_r,
                    "p_R_error": p_r,
                    "rho_q_error": rho_q,
                    "p_q_error": p_q,
                    "rho_R_q": rho_rq,
                    "partial_rho_J_error_given_R": partial_r,
                    "partial_p_J_error_given_R": partial_r_p,
                    "partial_rho_J_error_given_q": partial_q,
                    "partial_p_J_error_given_q": partial_q_p,
                    "partial_rho_J_error_given_R_q": partial_rq,
                    "partial_p_J_error_given_R_q": partial_rq_p,
                    "rank_R2_R": r2_r,
                    "rank_R2_q": r2_q,
                    "rank_R2_R_q": r2_rq,
                    "rank_R2_R_J": rank_r2(error, [rayleigh, gap]),
                    "rank_R2_q_J": rank_r2(error, [retention, gap]),
                    "rank_R2_R_q_J": rank_r2(
                        error, [rayleigh, retention, gap]
                    ),
                    "J_median": float(gap.median()),
                    "J_q25": float(gap.quantile(0.25)),
                    "J_q75": float(gap.quantile(0.75)),
                }
            )
    result = pd.DataFrame(rows)
    result["delta_rank_R2_J_over_R"] = (
        result["rank_R2_R_J"] - result["rank_R2_R"]
    )
    result["delta_rank_R2_J_over_q"] = (
        result["rank_R2_q_J"] - result["rank_R2_q"]
    )
    result["delta_rank_R2_J_over_R_q"] = (
        result["rank_R2_R_q_J"] - result["rank_R2_R_q"]
    )
    return result


def summarize(correlations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "rho_J_error",
        "rho_R_error",
        "rho_q_error",
        "partial_rho_J_error_given_R",
        "partial_rho_J_error_given_q",
        "partial_rho_J_error_given_R_q",
        "delta_rank_R2_J_over_R",
        "delta_rank_R2_J_over_q",
        "delta_rank_R2_J_over_R_q",
    ]

    def aggregate(columns: list[str]) -> pd.DataFrame:
        grouped = correlations.groupby(columns, as_index=False)
        pieces = []
        for metric in metrics:
            part = grouped[metric].agg(
                median="median",
                mean="mean",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
                minimum="min",
                maximum="max",
            )
            part.insert(len(columns), "metric", metric)
            pieces.append(part)
        return pd.concat(pieces, ignore_index=True)

    return (
        aggregate(["scope"]),
        aggregate(["scope", "task", "protocol", "degree"]),
        aggregate(["scope", "degree"]),
    )


def predictor_correlations(samples: pd.DataFrame) -> pd.DataFrame:
    physics = samples.drop_duplicates(
        ["protocol", "degree", "split", "geometry_id", "config_name"]
    )
    rows = []
    for (protocol, degree, task), cell in physics.groupby(
        ["protocol", "degree", "task"], sort=False
    ):
        for scope in ("all", "iid", "ood"):
            selected = scope_frame(cell, scope)
            pairs = {
                "R_q": ("initial_rayleigh_quotient", "target_norm_retention"),
                "R_J": ("initial_rayleigh_quotient", "jensen_gap"),
                "q_J": ("target_norm_retention", "jensen_gap"),
            }
            for name, (left, right) in pairs.items():
                rho, p = safe_spearman(selected[left], selected[right])
                rows.append(
                    {
                        "protocol": protocol,
                        "degree": int(degree),
                        "task": task,
                        "scope": scope,
                        "pair": name,
                        "n": len(selected),
                        "spearman_rho": rho,
                        "p_value": p,
                    }
                )
    return pd.DataFrame(rows)


def heatmap(correlations: pd.DataFrame) -> None:
    pooled = correlations[correlations["scope"] == "all"]
    matrix = pooled.pivot(index="task", columns="model", values="rho_J_error")
    matrix = matrix.reindex(index=TASKS, columns=MODELS)
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=30, ha="right")
    ax.set_yticks(range(len(TASKS)), TASKS)
    for row in range(len(TASKS)):
        for column in range(len(MODELS)):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.55 else "black")
    ax.set_title(r"Relative error vs. Jensen spectral gap $J_T$")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
    colorbar.set_label(r"Spearman $\rho_s(J_T,E_{rel})$")
    fig.tight_layout()
    fig.savefig(FIGURES / "jensen_relative_spearman_heatmap.png", dpi=240)
    plt.close(fig)


def task_bar(task_summary: pd.DataFrame) -> None:
    selected = task_summary[
        (task_summary["scope"] == "all")
        & (task_summary["metric"] == "rho_J_error")
    ].copy()
    selected["task"] = pd.Categorical(selected["task"], TASKS, ordered=True)
    selected = selected.sort_values("task")
    x = np.arange(len(selected))
    lower = selected["median"] - selected["q25"]
    upper = selected["q75"] - selected["median"]
    fig, ax = plt.subplots(figsize=(10.4, 4.5))
    colors = ["#3569a8" if degree < 2 else "#d16a3a" for degree in selected["degree"]]
    ax.bar(x, selected["median"], color=colors, width=0.72)
    ax.errorbar(x, selected["median"], yerr=[lower, upper], fmt="none", ecolor="black", capsize=3)
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.set_xticks(x, selected["task"], rotation=35, ha="right")
    ax.set_ylabel(r"Median $\rho_s(J_T,E_{rel})$ across models")
    ax.set_title("Task-selective effect of spectral broadening")
    fig.tight_layout()
    fig.savefig(FIGURES / "jensen_relative_by_task.png", dpi=240)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(SOURCE)
    samples["jensen_gap"] = (
        np.log(samples["target_norm_retention"])
        + KAPPA_T * samples["initial_rayleigh_quotient"]
    )
    samples["spectral_variance_proxy"] = samples["jensen_gap"] / (KAPPA_T**2)

    correlations = cell_correlations(samples)
    overall, by_task, by_degree = summarize(correlations)
    predictor = predictor_correlations(samples)

    sample_columns = [
        "model", "protocol", "degree", "split", "geometry_id", "config_name",
        "E_final", "initial_rayleigh_quotient", "target_norm_retention",
        "jensen_gap", "spectral_variance_proxy",
    ]
    samples[sample_columns].to_csv(
        SUPPORT / "jensen_relative_sample_data.csv", index=False
    )
    correlations.to_csv(SUPPORT / "jensen_relative_correlations.csv", index=False)
    overall.to_csv(SUPPORT / "jensen_relative_summary.csv", index=False)
    by_task.to_csv(SUPPORT / "jensen_relative_by_task.csv", index=False)
    by_degree.to_csv(SUPPORT / "jensen_relative_by_degree.csv", index=False)
    predictor.to_csv(SUPPORT / "spectral_predictor_correlations.csv", index=False)

    heatmap(correlations)
    task_bar(by_task)

    pooled = correlations[correlations["scope"] == "all"]
    qc = {
        "rows": int(len(samples)),
        "model_task_cells": int(len(pooled)),
        "minimum_jensen_gap": float(samples["jensen_gap"].min()),
        "nonnegative_jensen_gap": bool((samples["jensen_gap"] >= -1e-10).all()),
        "median_rho_J_error": float(pooled["rho_J_error"].median()),
        "median_partial_rho_J_error_given_R": float(
            pooled["partial_rho_J_error_given_R"].median()
        ),
        "median_delta_rank_R2_J_over_R": float(
            pooled["delta_rank_R2_J_over_R"].median()
        ),
    }
    (SUPPORT / "jensen_relative_qc.json").write_text(
        json.dumps(qc, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
