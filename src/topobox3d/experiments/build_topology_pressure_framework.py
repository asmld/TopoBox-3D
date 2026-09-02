"""Build the task-pressure, topology-bias, and architecture-selection framework."""

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
OUT = ROOT / "results/04_hodge_subspace_architecture"
SUPPORT = OUT / "supporting_data"
FIGURES = OUT / "figures"
MODELS = ["MGN-lite", "RIGNO", "Transolver", "GNOT", "GAOT", "TNO"]
TASKS = [f"{protocol}-k{degree}" for protocol in "ABCD" for degree in (0, 1, 2)]
MODEL_COLORS = {
    "MGN-lite": "#4C78A8", "RIGNO": "#72B7B2", "Transolver": "#F58518",
    "GNOT": "#E45756", "GAOT": "#B279A2", "TNO": "#54A24B",
}


def robust_scale(values: pd.Series) -> float:
    scale = float(values.quantile(0.75) - values.quantile(0.25))
    return max(scale, float(values.std()), 1e-8)


def task_pressure() -> pd.DataFrame:
    samples = pd.read_csv(
        ROOT
        / "results/03_error_source_univariate_regression/supporting_data/"
        "sample_level_analysis.csv"
    ).drop_duplicates(
        ["protocol", "degree", "split", "geometry_id", "config_name"]
    )
    samples["jensen_gap"] = (
        np.log(samples["target_norm_retention"])
        + 0.1 * samples["initial_rayleigh_quotient"]
    )
    scales = {
        variable: robust_scale(samples[variable])
        for variable in (
            "rho_H", "initial_rayleigh_quotient", "jensen_gap",
            "log_lambda1_positive", "log_nk",
        )
    }
    rows = []
    for (protocol, degree, task), cell in samples.groupby(
        ["protocol", "degree", "task"], sort=False
    ):
        iid = cell[cell["split"].eq("test_iid")]
        ood = cell[cell["split"].eq("test_ood")]
        beta_iid = iid["beta_k"].to_numpy(dtype=float)
        beta_ood = ood["beta_k"].to_numpy(dtype=float)
        beta_w1 = stats.wasserstein_distance(beta_iid, beta_ood)
        rho_w1 = stats.wasserstein_distance(iid["rho_H"], ood["rho_H"])
        kernel_pressure = np.sqrt(
            (beta_w1 / 3.0) ** 2 + (rho_w1 / scales["rho_H"]) ** 2
        )
        spectral_shifts = {}
        for variable in (
            "initial_rayleigh_quotient", "jensen_gap",
            "log_lambda1_positive", "log_nk",
        ):
            spectral_shifts[variable] = stats.wasserstein_distance(
                iid[variable], ood[variable]
            ) / scales[variable]
        spectrum_shift = float(
            np.sqrt(np.mean(np.square(list(spectral_shifts.values()))))
        )
        rows.append(
            {
                "task": task,
                "protocol": protocol,
                "degree": int(degree),
                "beta_k_iid_median": float(iid["beta_k"].median()),
                "beta_k_iid_min": float(iid["beta_k"].min()),
                "beta_k_iid_max": float(iid["beta_k"].max()),
                "beta_k_ood_median": float(ood["beta_k"].median()),
                "beta_k_wasserstein": float(beta_w1),
                "rho_H_wasserstein": float(rho_w1),
                "kernel_shift_pressure": float(kernel_pressure),
                "rayleigh_ood_median": float(
                    ood["initial_rayleigh_quotient"].median()
                ),
                "jensen_gap_ood_median": float(ood["jensen_gap"].median()),
                "spectrum_shift_pressure": spectrum_shift,
                **{
                    f"shift_{key}": float(value)
                    for key, value in spectral_shifts.items()
                },
            }
        )
    result = pd.DataFrame(rows)
    for variable in ("rayleigh_ood_median", "jensen_gap_ood_median"):
        result[f"percentile_{variable}"] = result[variable].rank(pct=True)
    result["positive_spectrum_load"] = np.sqrt(
        (
            result["percentile_rayleigh_ood_median"] ** 2
            + result["percentile_jensen_gap_ood_median"] ** 2
        ) / 2.0
    )
    result["kernel_pressure_level"] = pd.cut(
        result["kernel_shift_pressure"],
        bins=[-np.inf, 0.1, 0.5, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    result["spectrum_load_level"] = pd.cut(
        result["positive_spectrum_load"],
        bins=[-np.inf, 0.45, 0.70, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    return result


def controlled_probe_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pure = pd.concat(
        [
            pd.read_csv(SUPPORT / "pure_harmonic_controlled_torch.csv"),
            pd.read_csv(SUPPORT / "pure_harmonic_controlled_rigno.csv"),
        ],
        ignore_index=True,
    )
    summary = (
        pure.groupby(["model", "task", "protocol", "degree", "split"], as_index=False)
        .agg(
            geometries=("geometry_id", "nunique"),
            identity_error_median=("pure_harmonic_identity_error", "median"),
            identity_error_mean=("pure_harmonic_identity_error", "mean"),
            harmonic_preservation_median=("harmonic_preservation_error", "median"),
            harmonic_to_nonharmonic_leakage_median=(
                "harmonic_to_nonharmonic_leakage", "median"
            ),
        )
    )
    return pure, summary


def topology_bias_diagnostics(probe_summary: pd.DataFrame) -> pd.DataFrame:
    leakage = pd.read_csv(SUPPORT / "nonharmonic_to_harmonic_leakage.csv")
    profile = pd.read_csv(SUPPORT / "model_capability_profile.csv")
    ood = probe_summary[probe_summary["split"].eq("test_ood")]
    iid = probe_summary[probe_summary["split"].eq("test_iid")]
    ood_model = ood.groupby("model", as_index=False).agg(
        pure_H_identity_error=("identity_error_median", "median"),
        H_to_perp_leakage=("harmonic_to_nonharmonic_leakage_median", "median"),
        H_preservation_error=("harmonic_preservation_median", "median"),
    )
    iid_model = iid.groupby("model", as_index=False).agg(
        pure_H_identity_error_iid=("identity_error_median", "median"),
        H_to_perp_leakage_iid=(
            "harmonic_to_nonharmonic_leakage_median", "median"
        ),
    )
    p_to_h = leakage.groupby("model", as_index=False).agg(
        perp_to_H_leakage=("ood_leakage_mean", "median"),
        perp_to_H_leakage_iid=("iid_leakage_mean", "median"),
    )
    result = (
        ood_model.merge(iid_model, on="model")
        .merge(p_to_h, on="model")
        .merge(profile, on="model")
    )
    result["pure_H_identity_ood_iid"] = (
        result["pure_H_identity_error"]
        / result["pure_H_identity_error_iid"]
    )
    result["bidirectional_leakage"] = np.sqrt(
        result["H_to_perp_leakage"] * result["perp_to_H_leakage"]
    )
    access = {
        "TNO": "incidence + explicit harmonic basis",
        "MGN-lite": "raw graph incidence",
        "RIGNO": "coordinate-reconstructed graph",
        "GAOT": "coordinate-reconstructed graph",
        "Transolver": "coordinate attention",
        "GNOT": "coordinate attention",
    }
    result["structural_access"] = result["model"].map(access)
    result["hard_kernel_identity_guarantee"] = False
    for metric in (
        "pure_H_identity_error", "bidirectional_leakage",
        "median_ood_nonharmonic_relative",
    ):
        result[f"rank_{metric}"] = result[metric].rank(method="average")
    result["empirical_kernel_capability_rank"] = (
        result["rank_pure_H_identity_error"]
        + result["rank_bidirectional_leakage"]
    ) / 2.0
    result["positive_spectrum_filter_rank"] = result[
        "rank_median_ood_nonharmonic_relative"
    ]
    return result.sort_values("empirical_kernel_capability_rank")


def architecture_selection(
    pressure: pd.DataFrame, suitability: pd.DataFrame
) -> pd.DataFrame:
    winners = suitability[suitability["rank"].eq(1)][
        ["task", "model", "ood_loss"]
    ].rename(columns={"model": "observed_best_model"})
    table = pressure.merge(winners, on="task")
    recommendations = []
    for row in table.itertuples():
        kernel = row.kernel_pressure_level
        spectrum = row.spectrum_load_level
        if row.degree == 0:
            recommendation = "local message passing (MGN-like)"
        elif kernel == "high" and spectrum == "high":
            recommendation = (
                "hard kernel projector + multiscale positive-spectrum filter"
            )
        elif kernel == "high":
            recommendation = "kernel-aware cochain operator with identity branch"
        elif spectrum == "high":
            recommendation = "multiscale graph/attention spectral filter"
        else:
            recommendation = "lightweight graph or coordinate operator"
        recommendations.append(recommendation)
    table["recommended_architecture_class"] = recommendations
    return table


def pressure_validation(pressure: pd.DataFrame) -> pd.DataFrame:
    """Check what each pressure coordinate predicts without collapsing the axes."""
    rows = []

    def add_correlations(frame, unit, responses, predictors):
        for response in responses:
            for predictor in predictors:
                cell = frame[[response, predictor]].dropna()
                rho, p_value = stats.spearmanr(cell[predictor], cell[response])
                rows.append(
                    {
                        "analysis_unit": unit,
                        "response": response,
                        "pressure_coordinate": predictor,
                        "n": int(len(cell)),
                        "spearman_rho": float(rho),
                        "p_value": float(p_value),
                    }
                )

    seed_losses = pd.read_csv(
        ROOT
        / "results/01_model_task_training_results/supporting_data/"
        "seed_level_losses.csv"
    )
    seed_losses["task"] = (
        seed_losses["protocol"] + "-k" + seed_losses["degree"].astype(str)
    )
    model_task = seed_losses.groupby(["model", "task"], as_index=False).agg(
        overall_ood_error=("ood_loss", "median"),
        overall_ood_iid_ratio=("ood_iid_ratio", "median"),
    ).merge(pressure, on="task")
    predictors = [
        "kernel_shift_pressure", "positive_spectrum_load",
        "spectrum_shift_pressure",
    ]
    add_correlations(
        model_task, "model-task", ["overall_ood_error", "overall_ood_iid_ratio"],
        predictors,
    )

    task_level = model_task.groupby("task", as_index=False).agg(
        overall_ood_error=("overall_ood_error", "median"),
        overall_ood_iid_ratio=("overall_ood_iid_ratio", "median"),
        **{predictor: (predictor, "first") for predictor in predictors},
    )
    add_correlations(
        task_level, "task", ["overall_ood_error", "overall_ood_iid_ratio"],
        predictors,
    )

    component = pd.read_csv(SUPPORT / "component_degradation.csv").merge(
        pressure[["task", *predictors]], on="task"
    )
    for metric in ("harmonic_relative", "nonharmonic_relative"):
        cell = component[component["metric"].eq(metric)].rename(
            columns={
                "ood_mean": f"{metric}_ood_error",
                "ood_iid_ratio": f"{metric}_ood_iid_ratio",
            }
        )
        add_correlations(
            cell, "model-task-component",
            [f"{metric}_ood_error", f"{metric}_ood_iid_ratio"], predictors,
        )
    return pd.DataFrame(rows)


def plot_task_pressure(selection: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for model in MODELS:
        cell = selection[selection["observed_best_model"].eq(model)]
        ax.scatter(
            cell["kernel_shift_pressure"], cell["positive_spectrum_load"],
            s=95, color=MODEL_COLORS[model], label=model, alpha=0.9,
            edgecolor="white", linewidth=0.8,
        )
    label_offsets = {
        "A-k0": (6, 7), "B-k0": (6, 7), "C-k0": (6, -14),
        "D-k0": (6, -9), "A-k1": (6, 6), "C-k1": (6, 6),
        "A-k2": (6, 5), "B-k2": (6, 13),
        "B-k1": (6, 5), "C-k2": (6, 5), "D-k1": (6, 5), "D-k2": (6, 5),
    }
    for row in selection.itertuples():
        ax.annotate(
            row.task,
            (row.kernel_shift_pressure, row.positive_spectrum_load),
            xytext=label_offsets[row.task], textcoords="offset points", fontsize=8,
        )
    ax.axvline(0.1, color="#777777", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(0.70, color="#777777", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Kernel-shift pressure $\Pi_H$")
    ax.set_ylabel(r"Positive-spectrum load $\Pi_+$")
    ax.set_title("Task pressure plane and observed best architecture")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "task_topology_pressure_plane.png", dpi=240)
    plt.close(fig)


def plot_topology_bias(diagnostics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 6.0))
    sizes = 1800 * diagnostics["bidirectional_leakage"] / diagnostics[
        "bidirectional_leakage"
    ].max() + 80
    for index, row in diagnostics.reset_index(drop=True).iterrows():
        ax.scatter(
            row["pure_H_identity_error"],
            row["median_ood_nonharmonic_relative"],
            s=sizes.iloc[index], color=MODEL_COLORS[row["model"]],
            alpha=0.72, edgecolor="white", linewidth=1.0,
        )
        ax.annotate(
            row["model"],
            (row["pure_H_identity_error"], row["median_ood_nonharmonic_relative"]),
            xytext=(6, 5), textcoords="offset points", fontsize=9,
        )
    ax.set_xlabel("Pure-harmonic identity error (lower is better)")
    ax.set_ylabel(r"Non-harmonic filter error $E_\perp$ (lower is better)")
    ax.set_title("Kernel recognition and positive-spectrum filtering are distinct")
    ax.grid(alpha=0.2)
    ax.text(
        0.98, 0.98, "Bubble size: bidirectional subspace leakage",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "model_topology_bias_capabilities.png", dpi=240)
    plt.close(fig)


def plot_pure_harmonic_heatmap(summary: pd.DataFrame) -> None:
    matrix = summary[summary["split"].eq("test_ood")].pivot(
        index="task", columns="model", values="identity_error_median"
    ).reindex(columns=MODELS)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    image = ax.imshow(matrix.to_numpy(), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(MODELS)):
            value = matrix.iloc[row, column]
            ax.text(
                column, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value > 0.55 else "black",
            )
    ax.set_title("Inference-only pure-harmonic identity error (OOD)")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label(r"$\|\widehat S_Tw_H-w_H\|_M$")
    fig.tight_layout()
    fig.savefig(FIGURES / "pure_harmonic_identity_error.png", dpi=240)
    plt.close(fig)


def plot_pressure_validation(pressure: pd.DataFrame) -> None:
    """Visualize the three strongest pressure-validation relationships."""
    seed_losses = pd.read_csv(
        ROOT
        / "results/01_model_task_training_results/supporting_data/"
        "seed_level_losses.csv"
    )
    seed_losses["task"] = (
        seed_losses["protocol"] + "-k" + seed_losses["degree"].astype(str)
    )
    model_task = seed_losses.groupby(["model", "task"], as_index=False).agg(
        overall_ood_error=("ood_loss", "median"),
        overall_ood_iid_ratio=("ood_iid_ratio", "median"),
    ).merge(pressure, on="task")
    task_level = model_task.groupby("task", as_index=False).agg(
        overall_ood_error=("overall_ood_error", "median"),
        overall_ood_iid_ratio=("overall_ood_iid_ratio", "median"),
        positive_spectrum_load=("positive_spectrum_load", "first"),
        spectrum_shift_pressure=("spectrum_shift_pressure", "first"),
        degree=("degree", "first"),
    )
    component = pd.read_csv(SUPPORT / "component_degradation.csv").merge(
        pressure[["task", "kernel_shift_pressure", "spectrum_shift_pressure"]],
        on="task",
    )
    nonharmonic = component[component["metric"].eq("nonharmonic_relative")]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    axes = axes.ravel()
    degree_colors = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}

    def trend(ax, x, y):
        slope, intercept = np.polyfit(np.asarray(x), np.asarray(y), 1)
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(grid, intercept + slope * grid, color="#666666", lw=1.2,
                linestyle="--", zorder=1)

    ax = axes[0]
    offsets_a = {
        "A-k0": (4, -12), "B-k0": (4, 7), "C-k0": (4, -11),
        "D-k0": (4, 8), "A-k1": (4, 7), "B-k1": (4, 7),
        "C-k1": (4, -12), "D-k1": (4, 7), "A-k2": (4, -12),
        "B-k2": (4, 7), "C-k2": (4, 7), "D-k2": (4, 7),
    }
    for row in task_level.itertuples():
        ax.scatter(row.positive_spectrum_load, row.overall_ood_error, s=55,
                   color=degree_colors[row.degree], edgecolor="white", lw=0.7,
                   zorder=2)
        ax.annotate(row.task, (row.positive_spectrum_load, row.overall_ood_error),
                    xytext=offsets_a[row.task], textcoords="offset points", fontsize=7)
    trend(ax, task_level["positive_spectrum_load"], task_level["overall_ood_error"])
    ax.set_xlabel(r"Positive-spectrum load $\Pi_+$")
    ax.set_ylabel("Task-median OOD error")
    ax.set_title(r"(a) Absolute difficulty: $\rho_s=0.909$, $p=4.2\times10^{-5}$")
    degree_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=degree_colors[k],
                   label=f"k={k}", markersize=5)
        for k in (0, 1, 2)
    ]
    ax.legend(handles=degree_handles, frameon=False, fontsize=7, ncol=3,
              loc="upper left")

    ax = axes[1]
    offsets_b = {
        "A-k0": (4, 5), "B-k0": (4, 5), "C-k0": (4, 5),
        "D-k0": (4, 5), "A-k1": (4, -12), "B-k1": (4, -12),
        "C-k1": (4, 5), "D-k1": (-18, 5), "A-k2": (4, 5),
        "B-k2": (4, 5), "C-k2": (4, 5), "D-k2": (4, -12),
    }
    for row in task_level.itertuples():
        ax.scatter(row.spectrum_shift_pressure, row.overall_ood_iid_ratio, s=55,
                   color=degree_colors[row.degree], edgecolor="white", lw=0.7,
                   zorder=2)
        ax.annotate(row.task, (row.spectrum_shift_pressure, row.overall_ood_iid_ratio),
                    xytext=offsets_b[row.task], textcoords="offset points", fontsize=7)
    trend(ax, task_level["spectrum_shift_pressure"], task_level["overall_ood_iid_ratio"])
    ax.axhline(1.0, color="#999999", lw=0.8)
    ax.set_xlabel(r"Positive-spectrum shift $\Delta\Pi_+$")
    ax.set_ylabel("Task-median OOD/IID")
    ax.set_title(r"(b) Generalization shift: $\rho_s=0.608$, $p=0.036$")

    ax = axes[2]
    for model in MODELS:
        cell = nonharmonic[nonharmonic["model"].eq(model)]
        ax.scatter(cell["spectrum_shift_pressure"], cell["ood_iid_ratio"],
                   s=42, color=MODEL_COLORS[model], label=model, alpha=0.8,
                   edgecolor="white", lw=0.5, zorder=2)
    trend(ax, nonharmonic["spectrum_shift_pressure"], nonharmonic["ood_iid_ratio"])
    ax.axhline(1.0, color="#999999", lw=0.8)
    ax.set_xlabel(r"Positive-spectrum shift $\Delta\Pi_+$")
    ax.set_ylabel(r"Non-harmonic $E_\perp^{OOD}/E_\perp^{IID}$")
    ax.set_title(r"(c) Positive branch: $\rho_s=0.800$, $p=4.8\times10^{-9}$")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[3]
    harmonic = component[component["metric"].eq("harmonic_relative")]
    series = [
        (harmonic, "Harmonic", "#4C78A8", "o", -0.012),
        (nonharmonic, "Non-harmonic", "#E45756", "^", 0.012),
    ]
    for cell, label, color, marker, jitter in series:
        x = cell["kernel_shift_pressure"].to_numpy() + jitter
        ax.scatter(x, cell["ood_iid_ratio"], s=38, color=color, marker=marker,
                   alpha=0.62, edgecolor="white", lw=0.5, label=label, zorder=2)
        medians = cell.assign(
            kernel_group=cell["kernel_shift_pressure"].gt(0.1)
        ).groupby("kernel_group").agg(
            x=("kernel_shift_pressure", "median"),
            y=("ood_iid_ratio", "median"),
        )
        ax.plot(medians["x"] + jitter, medians["y"], color=color, marker=marker,
                lw=1.4, markersize=6, zorder=3)
    ax.axhline(1.0, color="#999999", lw=0.8)
    ax.set_xlabel(r"Kernel-shift pressure $\Pi_H$")
    ax.set_ylabel("Component OOD/IID")
    ax.set_title(
        r"(d) Kernel shift: $\rho_s(H)=0.148$ (n.s.), "
        r"$\rho_s(\perp)=0.504$ ($p=0.0017$)"
    )
    ax.legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "task_pressure_validation.png", dpi=240)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pressure = task_pressure()
    pure, probe_summary = controlled_probe_data()
    diagnostics = topology_bias_diagnostics(probe_summary)
    suitability = pd.read_csv(SUPPORT / "model_task_suitability.csv")
    selection = architecture_selection(pressure, suitability)
    validation = pressure_validation(pressure)

    pure.to_csv(SUPPORT / "pure_harmonic_controlled_all_models.csv", index=False)
    probe_summary.to_csv(SUPPORT / "pure_harmonic_controlled_summary.csv", index=False)
    pressure.to_csv(SUPPORT / "task_topology_pressure.csv", index=False)
    diagnostics.to_csv(SUPPORT / "topology_inductive_bias_diagnostics.csv", index=False)
    selection.to_csv(SUPPORT / "architecture_selection_map.csv", index=False)
    validation.to_csv(SUPPORT / "task_pressure_validation.csv", index=False)

    plot_task_pressure(selection)
    plot_topology_bias(diagnostics)
    plot_pure_harmonic_heatmap(probe_summary)
    plot_pressure_validation(pressure)

    qc = {
        "controlled_probe_rows": int(len(pure)),
        "models": int(pure["model"].nunique()),
        "tasks": int(pure["task"].nunique()),
        "splits": sorted(pure["split"].unique().tolist()),
        "median_pure_harmonic_ood_identity_by_model": (
            probe_summary[probe_summary["split"].eq("test_ood")]
            .groupby("model")["identity_error_median"].median().to_dict()
        ),
        "tno_has_hard_kernel_identity_guarantee": False,
        "pressure_validation": {
            f"{row.analysis_unit}:{row.response}:{row.pressure_coordinate}": {
                "n": int(row.n),
                "spearman_rho": float(row.spearman_rho),
                "p_value": float(row.p_value),
            }
            for row in validation.itertuples()
        },
    }
    (SUPPORT / "topology_framework_qc.json").write_text(
        json.dumps(qc, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
