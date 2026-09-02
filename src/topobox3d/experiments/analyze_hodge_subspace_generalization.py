"""Hodge-subspace diagnostics and model--task suitability analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path.cwd()
RUNS = ROOT / "runs/topobox3d"
TRAINING_RESULTS = ROOT / "results/01_model_task_training_results/supporting_data"
OUT = ROOT / "results/04_hodge_subspace_architecture"
SUPPORT = OUT / "supporting_data"
FIGURES = OUT / "figures"
PHYSICS_SOURCE = (
    ROOT
    / "results/03_error_source_univariate_regression/supporting_data/sample_level_analysis.csv"
)
MODELS_ON_DISK = ["MGN-lite", "rigno", "Transolver", "GNOT", "GAOT", "TNO"]
MODELS = ["MGN-lite", "RIGNO", "Transolver", "GNOT", "GAOT", "TNO"]
MODEL_LABEL = {"rigno": "RIGNO"}
PROTOCOLS = list("ABCD")
DEGREES = [0, 1, 2]
SEEDS = [0, 1, 2]
POSITIVE_CONFIGS = ["weak_harmonic", "balanced", "strong_harmonic"]
VALID_HARMONIC_TASKS = {
    ("A", 1), ("A", 2), ("B", 1), ("C", 2), ("D", 1), ("D", 2)
}
TASKS = [f"{protocol}-k{degree}" for protocol in PROTOCOLS for degree in DEGREES]
RNG = np.random.default_rng(20270803)


def read_prediction_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for disk_model in MODELS_ON_DISK:
        model = MODEL_LABEL.get(disk_model, disk_model)
        for protocol in PROTOCOLS:
            for degree in DEGREES:
                for seed in SEEDS:
                    run = (
                        RUNS / disk_model / f"protocol_{protocol}"
                        / f"k{degree}" / f"seed_{seed}"
                    )
                    for split in ("test_iid", "test_ood"):
                        path = run / f"{split}.jsonl"
                        with path.open(encoding="utf-8") as handle:
                            for line in handle:
                                if not line.strip():
                                    continue
                                item = json.loads(line)
                                coefficients = np.asarray(
                                    item.get("harmonic_coefficient_error", []),
                                    dtype=float,
                                )
                                fractions = item["realized_energy_fractions"]
                                rows.append(
                                    {
                                        "model": model,
                                        "protocol": protocol,
                                        "degree": degree,
                                        "task": f"{protocol}-k{degree}",
                                        "seed": seed,
                                        "split": split,
                                        "geometry_id": item["geometry_id"],
                                        "config_name": item["config_name"],
                                        "beta1": int(item["beta1"]),
                                        "beta2": int(item["beta2"]),
                                        "rho_H_json": float(fractions[2]),
                                        "relative_l2": float(item["relative_l2"]),
                                        "harmonic_relative": float(
                                            item["harmonic_relative"]
                                        ),
                                        "nonharmonic_relative": float(
                                            item["nonharmonic_relative"]
                                        ),
                                        "harmonic_coefficient_error_norm": float(
                                            np.linalg.norm(coefficients)
                                        ),
                                        "harmonic_dimension": len(coefficients),
                                    }
                                )
    return pd.DataFrame(rows)


def add_physical_scales(predictions: pd.DataFrame) -> pd.DataFrame:
    physics = pd.read_csv(PHYSICS_SOURCE).drop_duplicates(
        ["protocol", "degree", "split", "geometry_id", "config_name"]
    )
    physics = physics[
        [
            "protocol", "degree", "split", "geometry_id", "config_name",
            "rho_H", "target_norm_retention", "initial_rayleigh_quotient",
            "log_nk", "beta_k", "beta_total",
        ]
    ]
    merged = predictions.merge(
        physics,
        on=["protocol", "degree", "split", "geometry_id", "config_name"],
        how="left",
        validate="many_to_one",
    )
    if merged["target_norm_retention"].isna().any():
        raise RuntimeError("Some predictions could not be aligned to physics data")

    merged["target_harmonic_norm"] = np.sqrt(merged["rho_H"].clip(lower=0.0))
    merged["target_nonharmonic_norm"] = np.sqrt(
        (merged["target_norm_retention"] ** 2 - merged["rho_H"]).clip(lower=0.0)
    )
    merged["absolute_harmonic_error"] = merged[
        "harmonic_coefficient_error_norm"
    ]
    merged["absolute_nonharmonic_error"] = (
        merged["nonharmonic_relative"] * merged["target_nonharmonic_norm"]
    )
    merged["absolute_total_error"] = (
        merged["relative_l2"] * merged["target_norm_retention"]
    )
    component_energy = (
        merged["absolute_harmonic_error"] ** 2
        + merged["absolute_nonharmonic_error"] ** 2
    )
    merged["harmonic_error_energy_share"] = np.divide(
        merged["absolute_harmonic_error"] ** 2,
        component_energy,
        out=np.zeros(len(merged), dtype=float),
        where=component_energy.to_numpy() > 0,
    )
    merged["orthogonal_reconstruction_error"] = np.abs(
        merged["absolute_total_error"] ** 2 - component_energy
    ) / (merged["absolute_total_error"] ** 2 + 1e-12)
    merged["pure_nonharmonic_leakage"] = np.where(
        merged["config_name"].eq("non_harmonic"),
        merged["absolute_harmonic_error"],
        np.nan,
    )
    merged["leakage_per_sqrt_dimension"] = np.divide(
        merged["pure_nonharmonic_leakage"],
        np.sqrt(merged["harmonic_dimension"]),
        out=np.full(len(merged), np.nan),
        where=merged["harmonic_dimension"].to_numpy() > 0,
    )
    return merged


def seed_median(data: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model", "protocol", "degree", "task", "split", "geometry_id",
        "config_name", "beta1", "beta2", "rho_H", "target_norm_retention",
        "target_harmonic_norm", "target_nonharmonic_norm", "harmonic_dimension",
    ]
    values = [
        "relative_l2", "harmonic_relative", "nonharmonic_relative",
        "absolute_harmonic_error", "absolute_nonharmonic_error",
        "absolute_total_error", "harmonic_error_energy_share",
        "pure_nonharmonic_leakage", "leakage_per_sqrt_dimension",
        "orthogonal_reconstruction_error",
    ]
    return data.groupby(keys, as_index=False, dropna=False)[values].median()


def geometry_values(
    samples: pd.DataFrame,
    metric: str,
    positive_only: bool = True,
    require_harmonic_target: bool = False,
) -> pd.DataFrame:
    selected = samples
    if positive_only:
        selected = selected[selected["config_name"].isin(POSITIVE_CONFIGS)]
    if require_harmonic_target:
        selected = selected[selected["rho_H"] > 1e-12]
    return (
        selected.groupby(
            ["model", "protocol", "degree", "task", "split", "geometry_id"],
            as_index=False,
        )[metric]
        .mean()
    )


def bootstrap_ratio(iid: np.ndarray, ood: np.ndarray, repeats: int = 2000):
    iid_draw = RNG.integers(0, len(iid), size=(repeats, len(iid)))
    ood_draw = RNG.integers(0, len(ood), size=(repeats, len(ood)))
    ratios = ood[ood_draw].mean(axis=1) / (iid[iid_draw].mean(axis=1) + 1e-12)
    return tuple(float(value) for value in np.quantile(ratios, [0.025, 0.975]))


def component_degradation(samples: pd.DataFrame) -> pd.DataFrame:
    metric_specs = {
        "harmonic_relative": (True, True),
        "nonharmonic_relative": (True, False),
        "absolute_harmonic_error": (True, False),
        "absolute_nonharmonic_error": (True, False),
        "relative_l2": (True, False),
    }
    rows = []
    valid = samples[
        samples.apply(
            lambda row: (row["protocol"], int(row["degree"]))
            in VALID_HARMONIC_TASKS,
            axis=1,
        )
    ]
    for metric, (positive_only, require_harmonic_target) in metric_specs.items():
        geometry = geometry_values(
            valid, metric, positive_only, require_harmonic_target
        )
        for keys, cell in geometry.groupby(
            ["model", "protocol", "degree", "task"], sort=False
        ):
            model, protocol, degree, task = keys
            iid = cell[cell["split"].eq("test_iid")][metric].to_numpy()
            ood = cell[cell["split"].eq("test_ood")][metric].to_numpy()
            low, high = bootstrap_ratio(iid, ood)
            rows.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "degree": int(degree),
                    "task": task,
                    "metric": metric,
                    "iid_mean": float(iid.mean()),
                    "ood_mean": float(ood.mean()),
                    "ood_iid_ratio": float(ood.mean() / (iid.mean() + 1e-12)),
                    "ratio_ci95_low": low,
                    "ratio_ci95_high": high,
                    "significant_degradation": low > 1.0,
                    "significant_improvement": high < 1.0,
                }
            )
    return pd.DataFrame(rows)


def component_ease(samples: pd.DataFrame) -> pd.DataFrame:
    valid = samples[
        samples["config_name"].isin(POSITIVE_CONFIGS)
        & (samples["rho_H"] > 1e-12)
        & samples.apply(
            lambda row: (row["protocol"], int(row["degree"]))
            in VALID_HARMONIC_TASKS,
            axis=1,
        )
    ].copy()
    valid["harmonic_over_nonharmonic_relative"] = valid["harmonic_relative"] / (
        valid["nonharmonic_relative"] + 1e-12
    )
    rows = []
    for keys, cell in valid.groupby(
        ["model", "protocol", "degree", "task", "split"], sort=False
    ):
        model, protocol, degree, task, split = keys
        rows.append(
            {
                "model": model,
                "protocol": protocol,
                "degree": int(degree),
                "task": task,
                "split": split,
                "samples": len(cell),
                "harmonic_relative_median": float(cell["harmonic_relative"].median()),
                "nonharmonic_relative_median": float(
                    cell["nonharmonic_relative"].median()
                ),
                "median_paired_harmonic_over_nonharmonic": float(
                    cell["harmonic_over_nonharmonic_relative"].median()
                ),
                "fraction_harmonic_easier": float(
                    (cell["harmonic_relative"] < cell["nonharmonic_relative"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_contrast(samples: pd.DataFrame) -> pd.DataFrame:
    selected = samples[
        samples["config_name"].isin(POSITIVE_CONFIGS)
        & (samples["rho_H"] > 1e-12)
        & samples.apply(
            lambda row: (row["protocol"], int(row["degree"]))
            in VALID_HARMONIC_TASKS,
            axis=1,
        )
    ]
    geometry = selected.groupby(
        ["model", "protocol", "degree", "task", "split", "geometry_id"],
        as_index=False,
    ).agg(
        harmonic_relative=("harmonic_relative", "mean"),
        nonharmonic_relative=("nonharmonic_relative", "mean"),
    )
    rows = []
    for keys, cell in geometry.groupby(
        ["model", "protocol", "degree", "task"], sort=False
    ):
        iid = cell[cell["split"].eq("test_iid")][
            ["harmonic_relative", "nonharmonic_relative"]
        ].to_numpy()
        ood = cell[cell["split"].eq("test_ood")][
            ["harmonic_relative", "nonharmonic_relative"]
        ].to_numpy()
        rh = ood[:, 0].mean() / (iid[:, 0].mean() + 1e-12)
        rn = ood[:, 1].mean() / (iid[:, 1].mean() + 1e-12)
        draws = []
        for _ in range(2000):
            iid_sample = iid[RNG.integers(0, len(iid), len(iid))]
            ood_sample = ood[RNG.integers(0, len(ood), len(ood))]
            draw_rh = ood_sample[:, 0].mean() / (iid_sample[:, 0].mean() + 1e-12)
            draw_rn = ood_sample[:, 1].mean() / (iid_sample[:, 1].mean() + 1e-12)
            draws.append(draw_rh / (draw_rn + 1e-12))
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "model": keys[0], "protocol": keys[1], "degree": int(keys[2]),
                "task": keys[3], "harmonic_degradation": float(rh),
                "nonharmonic_degradation": float(rn),
                "harmonic_over_nonharmonic_degradation": float(rh / rn),
                "contrast_ci95_low": float(low), "contrast_ci95_high": float(high),
                "significantly_more_harmonic_sensitive": bool(low > 1),
                "significantly_more_nonharmonic_sensitive": bool(high < 1),
            }
        )
    return pd.DataFrame(rows)


def error_energy_share(samples: pd.DataFrame) -> pd.DataFrame:
    valid = samples[
        samples["config_name"].isin(POSITIVE_CONFIGS)
        & (samples["rho_H"] > 1e-12)
        & samples.apply(
            lambda row: (row["protocol"], int(row["degree"]))
            in VALID_HARMONIC_TASKS,
            axis=1,
        )
    ].copy()
    rows = []
    for keys, cell in valid.groupby(
        ["model", "protocol", "degree", "task", "split"], sort=False
    ):
        h2 = np.square(cell["absolute_harmonic_error"]).sum()
        n2 = np.square(cell["absolute_nonharmonic_error"]).sum()
        rows.append(
            {
                "model": keys[0],
                "protocol": keys[1],
                "degree": int(keys[2]),
                "task": keys[3],
                "split": keys[4],
                "harmonic_error_energy_share": float(h2 / (h2 + n2 + 1e-20)),
                "target_harmonic_energy_fraction_mean": float(cell["rho_H"].mean()),
                "harmonic_error_share_over_target_share": float(
                    (h2 / (h2 + n2 + 1e-20)) / (cell["rho_H"].mean() + 1e-20)
                ),
            }
        )
    return pd.DataFrame(rows)


def leakage_summary(samples: pd.DataFrame) -> pd.DataFrame:
    selected = samples[
        samples["config_name"].eq("non_harmonic")
        & samples.apply(
            lambda row: (row["protocol"], int(row["degree"]))
            in VALID_HARMONIC_TASKS,
            axis=1,
        )
    ]
    rows = []
    for keys, cell in selected.groupby(
        ["model", "protocol", "degree", "task"], sort=False
    ):
        geometry = cell.groupby(["split", "geometry_id"], as_index=False)[
            "pure_nonharmonic_leakage"
        ].mean()
        iid = geometry[geometry["split"].eq("test_iid")][
            "pure_nonharmonic_leakage"
        ].to_numpy()
        ood = geometry[geometry["split"].eq("test_ood")][
            "pure_nonharmonic_leakage"
        ].to_numpy()
        low, high = bootstrap_ratio(iid, ood)
        iid_positive = cell[
            cell["split"].eq("test_iid") & (cell["harmonic_dimension"] > 0)
        ]
        ood_positive = cell[
            cell["split"].eq("test_ood") & (cell["harmonic_dimension"] > 0)
        ]
        rows.append(
            {
                "model": keys[0],
                "protocol": keys[1],
                "degree": int(keys[2]),
                "task": keys[3],
                "iid_leakage_mean": float(iid.mean()),
                "ood_leakage_mean": float(ood.mean()),
                "ood_iid_ratio": float(ood.mean() / (iid.mean() + 1e-12)),
                "iid_positive_dimension_leakage_mean": float(
                    iid_positive["pure_nonharmonic_leakage"].mean()
                ),
                "ood_positive_dimension_leakage_mean": float(
                    ood_positive["pure_nonharmonic_leakage"].mean()
                ),
                "positive_dimension_ood_iid_ratio": float(
                    ood_positive["pure_nonharmonic_leakage"].mean()
                    / (iid_positive["pure_nonharmonic_leakage"].mean() + 1e-12)
                ),
                "iid_leakage_per_sqrt_dimension_mean": float(
                    iid_positive["leakage_per_sqrt_dimension"].mean()
                ),
                "ood_leakage_per_sqrt_dimension_mean": float(
                    ood_positive["leakage_per_sqrt_dimension"].mean()
                ),
                "ratio_ci95_low": low,
                "ratio_ci95_high": high,
                "significant_degradation": low > 1.0,
                "significant_improvement": high < 1.0,
            }
        )
    return pd.DataFrame(rows)


def dimension_effects(samples: pd.DataFrame) -> pd.DataFrame:
    selected = samples[
        samples["config_name"].eq("non_harmonic")
        & samples.apply(
            lambda row: (row["protocol"], int(row["degree"]))
            in VALID_HARMONIC_TASKS,
            axis=1,
        )
    ]
    return (
        selected.groupby(
            ["model", "protocol", "degree", "task", "split", "harmonic_dimension"],
            as_index=False,
        )
        .agg(
            geometries=("geometry_id", "nunique"),
            leakage_mean=("pure_nonharmonic_leakage", "mean"),
            leakage_median=("pure_nonharmonic_leakage", "median"),
            leakage_per_sqrt_dimension_mean=(
                "leakage_per_sqrt_dimension", "mean"
            ),
        )
    )


def model_task_suitability() -> pd.DataFrame:
    ood = pd.read_csv(TRAINING_RESULTS / "ood_loss_table.csv").set_index("Model")
    rows = []
    for task in TASKS:
        values = ood[task].sort_values()
        protocol, degree_text = task.split("-k")
        degree = int(degree_text)
        support = (protocol, degree) in VALID_HARMONIC_TASKS
        if degree == 0:
            task_type = "scalar_local_diffusion"
        elif support:
            task_type = "nontrivial_harmonic_subspace"
        else:
            task_type = "higher_order_zero_harmonic"
        best = values.index[0]
        best_value = float(values.iloc[0])
        for model, value in values.items():
            rows.append(
                {
                    "task": task,
                    "protocol": protocol,
                    "degree": degree,
                    "task_type": task_type,
                    "harmonic_supported": support,
                    "model": model,
                    "ood_loss": float(value),
                    "rank": int(values.rank(method="min").loc[model]),
                    "relative_regret_to_best": float(value / best_value),
                    "best_model": best,
                }
            )
    return pd.DataFrame(rows)


def model_capability_profile(
    ease: pd.DataFrame,
    degradation: pd.DataFrame,
    shares: pd.DataFrame,
    leakage: pd.DataFrame,
) -> pd.DataFrame:
    ood_ease = ease[ease["split"].eq("test_ood")]
    relative = degradation.pivot_table(
        index=["model", "task"], columns="metric", values="ood_mean"
    ).reset_index()
    robustness = degradation.pivot_table(
        index=["model", "task"], columns="metric", values="ood_iid_ratio"
    ).reset_index()
    robustness = robustness.rename(
        columns={
            "harmonic_relative": "harmonic_relative_degradation",
            "nonharmonic_relative": "nonharmonic_relative_degradation",
        }
    )
    share_ood = shares[shares["split"].eq("test_ood")][
        ["model", "task", "harmonic_error_energy_share"]
    ]
    cells = (
        ood_ease.merge(relative, on=["model", "task"])
        .merge(robustness, on=["model", "task"])
        .merge(share_ood, on=["model", "task"])
        .merge(
            leakage[["model", "task", "ood_leakage_mean", "ood_iid_ratio"]].rename(
                columns={"ood_iid_ratio": "leakage_degradation"}
            ),
            on=["model", "task"],
        )
    )
    rank_metrics = {
        "harmonic_fidelity_rank": "harmonic_relative",
        "nonharmonic_filter_rank": "nonharmonic_relative",
        "leakage_rank": "ood_leakage_mean",
    }
    for rank_name, metric in rank_metrics.items():
        cells[rank_name] = cells.groupby("task")[metric].rank(method="average")
    summary = cells.groupby("model", as_index=False).agg(
        median_ood_harmonic_relative=("harmonic_relative", "median"),
        median_ood_nonharmonic_relative=("nonharmonic_relative", "median"),
        median_harmonic_over_nonharmonic=(
            "median_paired_harmonic_over_nonharmonic", "median"
        ),
        median_harmonic_degradation=("harmonic_relative_degradation", "median"),
        median_nonharmonic_degradation=(
            "nonharmonic_relative_degradation", "median"
        ),
        median_harmonic_error_energy_share=("harmonic_error_energy_share", "median"),
        median_nonharmonic_to_harmonic_leakage=("ood_leakage_mean", "median"),
        mean_harmonic_fidelity_rank=("harmonic_fidelity_rank", "mean"),
        mean_nonharmonic_filter_rank=("nonharmonic_filter_rank", "mean"),
        mean_leakage_rank=("leakage_rank", "mean"),
    )
    return summary.sort_values("mean_harmonic_fidelity_rank")


def capability_performance_association(
    degradation: pd.DataFrame,
    leakage: pd.DataFrame,
    suitability: pd.DataFrame,
) -> pd.DataFrame:
    ood_components = degradation.pivot_table(
        index=["model", "task"], columns="metric", values="ood_mean"
    ).reset_index()
    cells = (
        suitability[suitability["harmonic_supported"]]
        .merge(ood_components, on=["model", "task"])
        .merge(
            leakage[["model", "task", "ood_leakage_mean"]],
            on=["model", "task"],
        )
    )
    metrics = [
        "harmonic_relative", "nonharmonic_relative",
        "absolute_harmonic_error", "absolute_nonharmonic_error",
        "ood_leakage_mean",
    ]
    rows = []
    for task, cell in cells.groupby("task"):
        for metric in metrics:
            result = stats.spearmanr(cell["ood_loss"], cell[metric])
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "models": len(cell),
                    "spearman_with_overall_ood_loss": float(result.statistic),
                    "p_value": float(result.pvalue),
                }
            )
    return pd.DataFrame(rows)


def figure_ease_sensitivity(ease: pd.DataFrame, contrast: pd.DataFrame) -> None:
    ease_agg = (
        ease.groupby(["model", "split"])[
            "median_paired_harmonic_over_nonharmonic"
        ]
        .median()
        .unstack("split")
        .reindex(MODELS)
    )
    sensitivity = (
        contrast.groupby("model")["harmonic_over_nonharmonic_degradation"]
        .median().reindex(MODELS)
    )
    x = np.arange(len(MODELS))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    axes[0].bar(x - 0.18, ease_agg["test_iid"], width=0.36, label="IID")
    axes[0].bar(x + 0.18, ease_agg["test_ood"], width=0.36, label="OOD")
    axes[0].axhline(1, color="0.35", linewidth=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, MODELS, rotation=30, ha="right")
    axes[0].set_ylabel(r"Median $E_H/E_\perp$ (log scale)")
    axes[0].set_title("Harmonic component is easier")
    axes[0].legend(frameon=False)
    axes[1].bar(x, sensitivity)
    axes[1].axhline(1, color="0.35", linewidth=0.9)
    axes[1].set_xticks(x, MODELS, rotation=30, ha="right")
    axes[1].set_ylabel(r"Median $R_H/R_\perp$")
    axes[1].set_title("But relatively more topology-sensitive")
    fig.tight_layout()
    fig.savefig(FIGURES / "harmonic_ease_vs_ood_sensitivity.png", dpi=240)
    plt.close(fig)


def figure_error_share(shares: pd.DataFrame) -> None:
    ood = shares[shares["split"].eq("test_ood")]
    tasks = [task for task in TASKS if tuple([task[0], int(task[-1])]) in VALID_HARMONIC_TASKS]
    matrix = ood.pivot(index="task", columns="model", values="harmonic_error_energy_share")
    matrix = matrix.reindex(index=tasks, columns=MODELS)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    image = ax.imshow(matrix.to_numpy(), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=30, ha="right")
    ax.set_yticks(range(len(tasks)), tasks)
    for row in range(len(tasks)):
        for column in range(len(MODELS)):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{100*value:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.55 else "black")
    ax.set_title("Harmonic share of total squared prediction error (OOD)")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.84)
    colorbar.set_label("Harmonic error-energy share")
    fig.tight_layout()
    fig.savefig(FIGURES / "harmonic_error_energy_share.png", dpi=240)
    plt.close(fig)


def figure_suitability(suitability: pd.DataFrame) -> None:
    matrix = suitability.pivot(index="task", columns="model", values="relative_regret_to_best")
    matrix = matrix.reindex(index=TASKS, columns=MODELS)
    display = np.log10(matrix.to_numpy())
    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    image = ax.imshow(display, cmap="YlGnBu", vmin=0, vmax=max(1.0, np.nanmax(display)), aspect="auto")
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=30, ha="right")
    task_labels = []
    for task in TASKS:
        supported = bool(
            suitability[suitability["task"].eq(task)]["harmonic_supported"].iloc[0]
        )
        task_labels.append(f"{task} [H]" if supported else task)
    ax.set_yticks(range(len(TASKS)), task_labels)
    for row, task in enumerate(TASKS):
        task_data = suitability[suitability["task"].eq(task)].set_index("model")
        for column, model in enumerate(MODELS):
            rank = int(task_data.loc[model, "rank"])
            ax.text(column, row, f"#{rank}", ha="center", va="center", fontsize=8,
                    color="white" if display[row, column] > 0.58 else "black")
    ax.set_title("Model--task suitability: OOD rank (H = nontrivial harmonic subspace)")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.84)
    colorbar.set_label(r"$\log_{10}$(loss / best loss)")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_task_suitability.png", dpi=240)
    plt.close(fig)


def figure_dimension_scaling(dimensions: pd.DataFrame) -> None:
    selected = dimensions[dimensions["harmonic_dimension"] > 0]
    raw = selected.groupby(["model", "harmonic_dimension"])["leakage_mean"].median()
    normalized = selected.groupby(["model", "harmonic_dimension"])[
        "leakage_per_sqrt_dimension_mean"
    ].median()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for model in MODELS:
        if model not in raw.index.get_level_values(0):
            continue
        model_raw = raw.loc[model]
        model_norm = normalized.loc[model]
        axes[0].plot(model_raw.index, model_raw.values, marker="o", label=model)
        axes[1].plot(model_norm.index, model_norm.values, marker="o", label=model)
    for ax in axes:
        ax.set_xticks([1, 2, 3])
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_xlabel(r"Harmonic dimension $\beta_k$")
    axes[0].set_ylabel(r"Leakage $\|P_H\widehat S P_\perp w_0\|_M$")
    axes[0].set_title("Total leakage grows with nullspace dimension")
    axes[1].set_xlabel(r"Harmonic dimension $\beta_k$")
    axes[1].set_ylabel(r"Leakage normalized by $\sqrt{\beta_k}$")
    axes[1].set_title("Per-mode leakage is approximately stable")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "leakage_dimension_scaling.png", dpi=240)
    plt.close(fig)


def main() -> None:
    SUPPORT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw = add_physical_scales(read_prediction_rows())
    samples = seed_median(raw)
    degradation = component_degradation(samples)
    ease = component_ease(samples)
    contrast = sensitivity_contrast(samples)
    shares = error_energy_share(samples)
    leakage = leakage_summary(samples)
    dimensions = dimension_effects(samples)
    suitability = model_task_suitability()
    capability = model_capability_profile(ease, degradation, shares, leakage)
    association = capability_performance_association(
        degradation, leakage, suitability
    )

    sample_columns = [
        "model", "protocol", "degree", "task", "split", "geometry_id",
        "config_name", "beta1", "beta2", "rho_H", "harmonic_dimension",
        "target_norm_retention", "relative_l2", "harmonic_relative",
        "nonharmonic_relative", "absolute_harmonic_error",
        "absolute_nonharmonic_error", "absolute_total_error",
        "harmonic_error_energy_share", "pure_nonharmonic_leakage",
    ]
    samples[sample_columns].to_csv(
        SUPPORT / "sample_component_errors_seed_median.csv", index=False
    )
    degradation.to_csv(SUPPORT / "component_degradation.csv", index=False)
    ease.to_csv(SUPPORT / "component_relative_ease.csv", index=False)
    contrast.to_csv(SUPPORT / "subspace_sensitivity_contrast.csv", index=False)
    shares.to_csv(SUPPORT / "component_error_energy_share.csv", index=False)
    leakage.to_csv(SUPPORT / "nonharmonic_to_harmonic_leakage.csv", index=False)
    dimensions.to_csv(SUPPORT / "leakage_by_harmonic_dimension.csv", index=False)
    suitability.to_csv(SUPPORT / "model_task_suitability.csv", index=False)
    capability.to_csv(SUPPORT / "model_capability_profile.csv", index=False)
    association.to_csv(
        SUPPORT / "capability_performance_association.csv", index=False
    )

    figure_ease_sensitivity(ease, contrast)
    figure_error_share(shares)
    figure_suitability(suitability)
    figure_dimension_scaling(dimensions)

    positive = ease[ease["split"].eq("test_ood")]
    wide = contrast.copy()
    target_winners = (
        suitability[suitability["rank"].eq(1)]
        .groupby(["task_type", "model"]).size().rename("wins").reset_index()
    )
    qc = {
        "raw_seed_rows": int(len(raw)),
        "seed_median_rows": int(len(samples)),
        "maximum_rho_H_alignment_error": float(
            np.max(np.abs(raw["rho_H_json"] - raw["rho_H"]))
        ),
        "median_orthogonal_reconstruction_relative_error": float(
            raw["orthogonal_reconstruction_error"].median()
        ),
        "ood_cells_harmonic_easier_fraction_median": float(
            positive["fraction_harmonic_easier"].median()
        ),
        "median_paired_EH_over_EN_ood": float(
            positive["median_paired_harmonic_over_nonharmonic"].median()
        ),
        "harmonic_more_topology_sensitive_cells": int(
            (wide["harmonic_over_nonharmonic_degradation"] > 1).sum()
        ),
        "harmonic_significantly_more_sensitive_cells": int(
            wide["significantly_more_harmonic_sensitive"].sum()
        ),
        "topology_target_harmonic_more_sensitive_cells": int(
            (
                (wide["protocol"] != "A")
                & (wide["harmonic_over_nonharmonic_degradation"] > 1)
            ).sum()
        ),
        "topology_target_cells": int((wide["protocol"] != "A").sum()),
        "median_topology_target_harmonic_over_nonharmonic_degradation": float(
            wide[wide["protocol"] != "A"][
                "harmonic_over_nonharmonic_degradation"
            ].median()
        ),
        "valid_model_task_cells": int(len(wide)),
        "median_harmonic_error_energy_share_ood": float(
            shares[shares["split"].eq("test_ood")][
                "harmonic_error_energy_share"
            ].median()
        ),
        "median_positive_dimension_leakage_ratio_ood_iid": float(
            leakage["positive_dimension_ood_iid_ratio"].median()
        ),
        "median_leakage_per_sqrt_dimension_ratio_ood_iid": float(
            (
                leakage["ood_leakage_per_sqrt_dimension_mean"]
                / leakage["iid_leakage_per_sqrt_dimension_mean"]
            ).median()
        ),
        "task_type_winners": target_winners.to_dict("records"),
        "median_cross_model_spearman_overall_loss_vs_harmonic_error": float(
            association[association["metric"] == "harmonic_relative"][
                "spearman_with_overall_ood_loss"
            ].median()
        ),
        "median_cross_model_spearman_overall_loss_vs_nonharmonic_error": float(
            association[association["metric"] == "nonharmonic_relative"][
                "spearman_with_overall_ood_loss"
            ].median()
        ),
        "median_cross_model_spearman_overall_loss_vs_pure_nonharmonic_leakage": float(
            association[association["metric"] == "ood_leakage_mean"][
                "spearman_with_overall_ood_loss"
            ].median()
        ),
    }
    (SUPPORT / "qc_summary.json").write_text(
        json.dumps(qc, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
