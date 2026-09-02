"""Univariate error-source analysis for all baseline models and test tasks.

The response is E_final = relative_l2. Repeated seeds are aggregated by the
median for each model/protocol/degree/split/geometry/config observation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ("MGN-lite", "rigno", "Transolver", "GNOT", "GAOT", "TNO")
MODEL_LABELS = {"rigno": "RIGNO"}
PROTOCOLS = ("A", "B", "C", "D")
DEGREES = (0, 1, 2)
SPLITS = ("test_iid", "test_ood")
CONFIG_NAMES = (
    "non_harmonic",
    "weak_harmonic",
    "balanced",
    "strong_harmonic",
)
VARIABLES = (
    "log_nk",
    "beta_k",
    "beta_total",
    "rho_H",
    "log_lambda1_positive",
    "initial_rayleigh_quotient",
    "target_norm_retention",
)
VARIABLE_LABELS = {
    "log_nk": r"$\log n_k$",
    "beta_k": r"$\beta_k$",
    "beta_total": r"$\beta_1+\beta_2$",
    "rho_H": r"$\rho_H$",
    "log_lambda1_positive": r"$\log\lambda_1^+$",
    "initial_rayleigh_quotient": r"$\mathcal{R}(w_0)$",
    "target_norm_retention": r"$\|w_T\|_M/\|w_0\|_M$",
}
DISCRETE_VARIABLES = {"beta_k", "beta_total"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/03_error_source_univariate_regression"),
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def load_incidence(archive: np.lib.npyio.NpzFile, prefix: str) -> sparse.csr_matrix:
    shape = tuple(int(v) for v in archive[f"{prefix}_shape"])
    boundary = sparse.coo_matrix(
        (
            archive[f"{prefix}_value"].astype(np.float64),
            (archive[f"{prefix}_row"], archive[f"{prefix}_col"]),
        ),
        shape=shape,
    ).tocsr()
    return boundary.T.tocsr()


def rayleigh_quotients(
    w0: np.ndarray,
    degree: int,
    derivatives: tuple[sparse.csr_matrix, ...],
    masses: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Compute w^T K_k w / w^T M_k w without explicitly assembling K_k."""

    values = np.asarray(w0, dtype=np.float64)
    mass_k = masses[degree]
    denom = np.einsum("ij,j,ij->i", values, mass_k, values)

    upper_derivative = derivatives[degree]
    upper_values = upper_derivative @ values.T
    numerator = np.einsum(
        "ij,i,ij->j", upper_values, masses[degree + 1], upper_values
    )

    if degree > 0:
        weighted = (mass_k[None, :] * values).T
        lower_values = derivatives[degree - 1].T @ weighted
        numerator += np.einsum(
            "ij,i,ij->j",
            lower_values,
            1.0 / masses[degree - 1],
            lower_values,
        )
    return numerator / denom


def build_feature_table(root: Path) -> pd.DataFrame:
    index_path = root / "data/TopoBox-3D-HodgeHeat/index.csv"
    index = pd.read_csv(index_path)
    index = index[index["split"].isin(SPLITS)].copy()
    spectral = pd.read_csv(
        root / "results/comprehensive_analysis/dataset_spectral_diagnostics.csv"
    )
    spectral = spectral[spectral["split"].isin(SPLITS)].copy()
    spectral_lookup = spectral.set_index(
        ["geometry_id", "degree", "config"]
    )[["lambda_min_positive", "target_norm_retention"]]

    rows: list[dict[str, object]] = []
    pde_root = root / "data/TopoBox-3D-HodgeHeat"
    geometry_root = root / "data/TopoBox-3D"

    for shard_rel, shard_rows in index.groupby("shard", sort=True):
        with h5py.File(pde_root / shard_rel, "r") as h5:
            for record in shard_rows.itertuples(index=False):
                mesh_path = (
                    geometry_root
                    / f"protocol_{record.protocol}"
                    / record.split
                    / record.geometry_id
                    / "mesh.npz"
                )
                with np.load(mesh_path, allow_pickle=False) as mesh:
                    derivatives = (
                        load_incidence(mesh, "incidence_1"),
                        load_incidence(mesh, "incidence_2"),
                        load_incidence(mesh, "incidence_3"),
                    )
                    n_counts = (
                        int(mesh["points"].shape[0]),
                        int(mesh["edges"].shape[0]),
                        int(mesh["faces"].shape[0]),
                    )
                    mass3 = 1.0 / mesh["tetra_volumes"].astype(np.float64)

                geometry_group = h5[record.group]
                stored_masses = tuple(
                    np.asarray(geometry_group[f"k{k}"]["mass"], dtype=np.float64)
                    for k in DEGREES
                )
                masses = (*stored_masses, mass3)

                for degree in DEGREES:
                    degree_group = geometry_group[f"k{degree}"]
                    realized = np.asarray(
                        degree_group["realized_energy_fractions"], dtype=np.float64
                    )
                    rq = rayleigh_quotients(
                        np.asarray(degree_group["w0"], dtype=np.float64),
                        degree,
                        derivatives,
                        masses,
                    )
                    retention_h5 = np.asarray(
                        degree_group["relative_final_mass_norm"], dtype=np.float64
                    )
                    beta_k = (
                        1 if degree == 0
                        else int(record.beta1) if degree == 1
                        else int(record.beta2)
                    )
                    for config_index, config_name in enumerate(CONFIG_NAMES):
                        spectral_row = spectral_lookup.loc[
                            (record.geometry_id, degree, config_name)
                        ]
                        lambda1 = float(spectral_row["lambda_min_positive"])
                        retention_csv = float(spectral_row["target_norm_retention"])
                        retention = float(retention_h5[config_index])
                        if not np.isclose(
                            retention, retention_csv, rtol=1e-6, atol=1e-8
                        ):
                            raise ValueError(
                                f"Retention mismatch: {record.geometry_id} k{degree}"
                            )
                        rows.append(
                            {
                                "geometry_id": record.geometry_id,
                                "protocol": record.protocol,
                                "split": record.split,
                                "degree": degree,
                                "task": f"{record.protocol}-k{degree}",
                                "config_name": config_name,
                                "beta1": int(record.beta1),
                                "beta2": int(record.beta2),
                                "n_k": n_counts[degree],
                                "log_nk": math.log(n_counts[degree]),
                                "beta_k": beta_k,
                                "beta_total": int(record.beta1 + record.beta2),
                                "rho_H": float(realized[config_index, 2]),
                                "lambda1_positive": lambda1,
                                "log_lambda1_positive": math.log(lambda1),
                                "initial_rayleigh_quotient": float(rq[config_index]),
                                "target_norm_retention": retention,
                            }
                        )
    result = pd.DataFrame(rows)
    expected = len(PROTOCOLS) * len(SPLITS) * 200 * len(DEGREES) * len(CONFIG_NAMES)
    if len(result) != expected:
        raise ValueError(f"Expected {expected} feature rows, found {len(result)}")
    return result


def load_errors(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[pd.DataFrame] = []
    for model in MODELS:
        for protocol in PROTOCOLS:
            for degree in DEGREES:
                for seed in (0, 1, 2):
                    run_dir = (
                        root
                        / "results"
                        / model
                        / f"protocol_{protocol}"
                        / f"k{degree}"
                        / f"seed_{seed}"
                    )
                    for split in SPLITS:
                        path = run_dir / f"{split}.jsonl"
                        frame = pd.read_json(path, lines=True)
                        frame = frame[
                            [
                                "geometry_id",
                                "protocol",
                                "split",
                                "degree",
                                "config_name",
                                "relative_l2",
                            ]
                        ].copy()
                        frame["model_key"] = model
                        frame["model"] = MODEL_LABELS.get(model, model)
                        frame["seed"] = seed
                        seed_rows.append(frame)
    raw = pd.concat(seed_rows, ignore_index=True)
    keys = [
        "model_key",
        "model",
        "protocol",
        "degree",
        "split",
        "geometry_id",
        "config_name",
    ]
    counts = raw.groupby(keys, observed=True)["seed"].nunique()
    if not (counts == 3).all():
        raise ValueError("Some observations do not have exactly three seeds.")
    aggregated = (
        raw.groupby(keys, as_index=False, observed=True)["relative_l2"]
        .median()
        .rename(columns={"relative_l2": "E_final"})
    )
    aggregated["task"] = (
        aggregated["protocol"] + "-k" + aggregated["degree"].astype(str)
    )
    return raw, aggregated


def cluster_robust_regression(
    frame: pd.DataFrame, variable: str
) -> dict[str, object]:
    clean = frame[[variable, "E_final", "geometry_id"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    x = clean[variable].to_numpy(dtype=float)
    y = clean["E_final"].to_numpy(dtype=float)
    n = len(clean)
    n_unique = int(pd.Series(x).nunique())
    clusters = clean["geometry_id"].astype(str).to_numpy()
    n_clusters = int(pd.Series(clusters).nunique())
    base = {
        "n": n,
        "n_geometries": n_clusters,
        "n_unique_x": n_unique,
        "x_min": float(np.min(x)) if n else np.nan,
        "x_max": float(np.max(x)) if n else np.nan,
        "x_mean": float(np.mean(x)) if n else np.nan,
        "x_sd": float(np.std(x, ddof=1)) if n > 1 else np.nan,
        "y_mean": float(np.mean(y)) if n else np.nan,
        "y_median": float(np.median(y)) if n else np.nan,
        "y_sd": float(np.std(y, ddof=1)) if n > 1 else np.nan,
    }
    if n < 3 or n_unique < 2 or np.std(y) == 0:
        return {
            **base,
            "status": "constant_predictor" if n_unique < 2 else "insufficient_data",
            "intercept": np.nan,
            "slope": np.nan,
            "slope_se_cluster": np.nan,
            "slope_ci95_low": np.nan,
            "slope_ci95_high": np.nan,
            "p_cluster": np.nan,
            "standardized_slope": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    X = np.column_stack((np.ones(n), x))
    xtx_inv = np.linalg.inv(X.T @ X)
    coefficients = xtx_inv @ X.T @ y
    residuals = y - X @ coefficients
    meat = np.zeros((2, 2), dtype=float)
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        score = X[mask].T @ residuals[mask]
        meat += np.outer(score, score)
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - 2))
    covariance = correction * xtx_inv @ meat @ xtx_inv
    slope_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    dof = max(n_clusters - 1, 1)
    critical = float(stats.t.ppf(0.975, dof))
    slope = float(coefficients[1])
    t_value = slope / slope_se if slope_se > 0 else np.inf
    p_cluster = float(2 * stats.t.sf(abs(t_value), dof))
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    residual_ss = float(np.sum(residuals**2))
    return {
        **base,
        "status": "ok",
        "intercept": float(coefficients[0]),
        "slope": slope,
        "slope_se_cluster": slope_se,
        "slope_ci95_low": slope - critical * slope_se,
        "slope_ci95_high": slope + critical * slope_se,
        "p_cluster": p_cluster,
        "standardized_slope": slope * np.std(x, ddof=1) / np.std(y, ddof=1),
        "r2": 1.0 - residual_ss / total_ss,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def run_regressions(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["model", "protocol", "degree", "task"]
    for group_values, task_frame in data.groupby(group_columns, sort=False):
        identity = dict(zip(group_columns, group_values))
        for variable in VARIABLES:
            for scope, scoped in (
                ("all", task_frame),
                ("iid", task_frame[task_frame["split"] == "test_iid"]),
                ("ood", task_frame[task_frame["split"] == "test_ood"]),
            ):
                rows.append(
                    {
                        **identity,
                        "variable": variable,
                        "scope": scope,
                        **cluster_robust_regression(scoped, variable),
                    }
                )
    return pd.DataFrame(rows)


def make_scatter_plot(
    task_frame: pd.DataFrame,
    regression_row: pd.Series,
    variable: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    rng = np.random.default_rng(20260731)
    styles = {
        "test_iid": ("#2878B5", "o", "IID"),
        "test_ood": ("#E07A1F", "^", "OOD"),
    }
    for split, (color, marker, label) in styles.items():
        subset = task_frame[task_frame["split"] == split]
        x = subset[variable].to_numpy(dtype=float)
        if variable in DISCRETE_VARIABLES:
            x = x + rng.uniform(-0.07, 0.07, size=len(x))
        ax.scatter(
            x,
            subset["E_final"],
            s=9,
            alpha=0.22,
            color=color,
            marker=marker,
            linewidths=0,
            label=label,
            rasterized=True,
        )

    if regression_row["status"] == "ok":
        x_min = float(task_frame[variable].min())
        x_max = float(task_frame[variable].max())
        x_line = np.linspace(x_min, x_max, 100)
        y_line = regression_row["intercept"] + regression_row["slope"] * x_line
        ax.plot(x_line, y_line, color="#222222", linewidth=1.7, label="Pooled OLS")
        annotation = (
            rf"$\beta_{{std}}$={regression_row['standardized_slope']:.2f}"
            "\n"
            rf"$R^2$={regression_row['r2']:.2f}, "
            rf"$\rho_s$={regression_row['spearman_rho']:.2f}"
            "\n"
            rf"$p_{{cluster}}$={regression_row['p_cluster']:.2g}"
        )
    else:
        annotation = "Predictor constant\nwithin this task"

    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#CCCCCC"},
    )
    model = task_frame["model"].iloc[0]
    task = task_frame["task"].iloc[0]
    ax.set_title(f"{model} | {task}", fontsize=11)
    ax.set_xlabel(VARIABLE_LABELS[variable])
    ax.set_ylabel(r"$E_{\rm final}$ (relative $L^2$)")
    ax.grid(True, color="#DDDDDD", linewidth=0.55, alpha=0.65)
    ax.legend(frameon=False, fontsize=8, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def make_heatmaps(pooled: pd.DataFrame, figures_root: Path) -> None:
    task_order = [f"{p}-k{k}" for k in DEGREES for p in PROTOCOLS]
    model_order = [MODEL_LABELS.get(model, model) for model in MODELS]
    for variable in VARIABLES:
        selected = pooled[pooled["variable"] == variable]
        for metric, suffix, cmap, vmin, vmax in (
            ("standardized_slope", "standardized-slope", "coolwarm", -1.0, 1.0),
            ("spearman_rho", "spearman-rho", "coolwarm", -1.0, 1.0),
            ("r2", "r2", "YlGnBu", 0.0, 1.0),
        ):
            matrix = (
                selected.pivot(index="task", columns="model", values=metric)
                .reindex(index=task_order, columns=model_order)
            )
            fig, ax = plt.subplots(figsize=(8.2, 6.3), constrained_layout=True)
            image = ax.imshow(
                matrix.to_numpy(dtype=float),
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks(range(len(model_order)), model_order, rotation=35, ha="right")
            ax.set_yticks(range(len(task_order)), task_order)
            ax.set_title(f"{VARIABLE_LABELS[variable]} | pooled {metric}")
            for i in range(len(task_order)):
                for j in range(len(model_order)):
                    value = matrix.iloc[i, j]
                    ax.text(
                        j,
                        i,
                        "—" if pd.isna(value) else f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#111111",
                    )
            fig.colorbar(image, ax=ax, shrink=0.82)
            path = figures_root / "overview" / f"{variable}__{suffix}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180)
            plt.close(fig)


def write_tables(
    data: pd.DataFrame,
    regressions: pd.DataFrame,
    output: Path,
) -> None:
    support = output / "supporting_data"
    support.mkdir(parents=True, exist_ok=True)
    data.to_csv(support / "sample_level_analysis.csv", index=False)
    regressions.to_csv(support / "univariate_regression_long.csv", index=False)
    pooled = regressions[regressions["scope"] == "all"].copy()
    pooled.to_csv(support / "univariate_regression_pooled.csv", index=False)

    for metric in ("standardized_slope", "spearman_rho", "r2", "p_cluster"):
        matrix = pooled.pivot(
            index=["model", "task"], columns="variable", values=metric
        ).reset_index()
        matrix.to_csv(support / f"pooled_{metric}_matrix.csv", index=False)

    availability = (
        regressions.groupby(["protocol", "degree", "task", "variable", "scope"])
        .agg(
            n_unique_x_min=("n_unique_x", "min"),
            n_unique_x_max=("n_unique_x", "max"),
            estimable_models=("status", lambda x: int((x == "ok").sum())),
        )
        .reset_index()
    )
    availability.to_csv(support / "variable_availability.csv", index=False)

    estimable = regressions[regressions["status"] == "ok"].copy()
    aggregate = (
        estimable.groupby(["scope", "variable"], as_index=False)
        .agg(
            estimable_units=("r2", "size"),
            median_r2=("r2", "median"),
            mean_r2=("r2", "mean"),
            median_abs_standardized_slope=(
                "standardized_slope",
                lambda values: values.abs().median(),
            ),
            median_spearman_rho=("spearman_rho", "median"),
            significant_units=("p_cluster", lambda values: int((values < 0.05).sum())),
        )
    )
    aggregate["significant_fraction"] = (
        aggregate["significant_units"] / aggregate["estimable_units"]
    )
    aggregate.to_csv(support / "aggregate_variable_summary.csv", index=False)

    by_degree = (
        estimable.groupby(["scope", "degree", "variable"], as_index=False)
        .agg(
            estimable_units=("r2", "size"),
            median_r2=("r2", "median"),
            median_standardized_slope=("standardized_slope", "median"),
            median_spearman_rho=("spearman_rho", "median"),
            significant_units=("p_cluster", lambda values: int((values < 0.05).sum())),
        )
    )
    by_degree.to_csv(support / "aggregate_by_degree_variable.csv", index=False)

    pooled_ok = pooled[pooled["status"] == "ok"]
    top_indices = pooled_ok.groupby(["model", "task"])["r2"].idxmax()
    top_predictors = pooled_ok.loc[
        top_indices,
        [
            "model",
            "protocol",
            "degree",
            "task",
            "variable",
            "standardized_slope",
            "r2",
            "spearman_rho",
            "p_cluster",
        ],
    ].sort_values(["model", "degree", "protocol"])
    top_predictors.to_csv(support / "top_predictor_by_model_task.csv", index=False)

    figure_rows = []
    for model in [MODEL_LABELS.get(item, item) for item in MODELS]:
        model_slug = model.lower().replace("-", "_")
        for protocol in PROTOCOLS:
            for degree in DEGREES:
                task = f"{protocol}-k{degree}"
                for variable in VARIABLES:
                    figure_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "variable": variable,
                            "relative_path": (
                                f"figures/scatter/{variable}/"
                                f"{model_slug}__protocol_{protocol.lower()}__k{degree}.png"
                            ),
                        }
                    )
    pd.DataFrame(figure_rows).to_csv(support / "figure_index.csv", index=False)


def write_method_document(output: Path, regressions: pd.DataFrame) -> None:
    pooled = regressions[regressions["scope"] == "all"]
    constant = pooled[pooled["status"] != "ok"]
    lines = [
        "# 结果小节 3：测试误差的单变量来源分析",
        "",
        "> 当前状态：统计表与绘图已完成；因果与多变量结论暂不在本节中下定论。",
        "",
        "## 1. 分析口径",
        "",
        "- 响应变量：`E_final = relative_l2`。",
        "- 观测单位：模型 × Protocol × 阶数 × split × geometry × 初值配置。",
        "- 重复实验：同一观测的 3 个 seed 先取中位数，因此每个模型–任务共有 1,600 个点（IID 800，OOD 800）。",
        "- 每个解释变量单独拟合 `E_final = a + b x`；不在这一轮加入协变量或交互项。",
        "- 斜率显著性使用 geometry-cluster 稳健标准误，使同一几何的 4 个初值不被当成完全独立样本。",
        "- 同时报告原始斜率、标准化斜率、95% 区间、R²、Pearson r 与 Spearman ρ。",
        "",
        "## 2. 七个解释变量",
        "",
        "1. `log_nk`：当前 k 阶 token 数的自然对数；k=0/1/2 分别使用点/边/面数。",
        "2. `beta_k`：相关阶 Betti 数；k=0 时 beta_0=1。",
        "3. `beta_total`：beta_1+beta_2。",
        "4. `rho_H`：初值实际实现的 harmonic energy fraction。",
        "5. `log_lambda1_positive`：对应 Hodge Laplacian 最小正广义特征值的自然对数。",
        "6. `initial_rayleigh_quotient`：w0^T K_k w0 / (w0^T M_k w0)。",
        "7. `target_norm_retention`：||wT||_M / ||w0||_M。",
        "",
        "## 3. 解释边界",
        "",
        "- 这是逐变量描述性回归。变量之间存在相关性，斜率不能直接解释为因果效应。",
        "- 合并 IID 与 OOD 的斜率可能同时包含 split 差异；因此长表另附 IID-only 与 OOD-only 回归。",
        "- Betti 数为离散变量；散点图只在显示时加入轻微横向抖动，回归使用原始整数。",
        "- 若变量在某任务内为常数，结果标记为 `constant_predictor`，不报告斜率或相关系数。",
        "",
        "## 4. 不可估计单元",
        "",
        f"全样本口径共有 {len(constant)} / {len(pooled)} 个模型–任务–变量单元因解释变量恒定而不可估计。",
        "这些单元主要来自 k=0 的 beta_0=1、Protocol A 的固定 Betti 结构，以及 k=0 初值抑制常数调和模态时 rho_H=0。",
        "",
        "## 5. 支撑数据",
        "",
        "- `supporting_data/sample_level_analysis.csv`：种子中位数后的完整样本级分析表。",
        "- `supporting_data/univariate_regression_long.csv`：全样本、IID-only、OOD-only 的完整回归长表。",
        "- `supporting_data/univariate_regression_pooled.csv`：论文主绘图对应的全样本回归表。",
        "- `supporting_data/pooled_*_matrix.csv`：标准化斜率、Spearman、R²、聚类稳健 p 值矩阵。",
        "- `supporting_data/variable_availability.csv`：各变量在任务内是否具有足够变异。",
        "- `supporting_data/figure_index.csv`：504 张散点图的路径索引。",
        "- `figures/scatter/`：逐变量、逐模型、逐任务散点图。",
        "- `figures/overview/`：跨模型与任务的标准化斜率、Spearman 和 R² 热图。",
        "",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    features = build_feature_table(root)
    raw_errors, errors = load_errors(root)
    merge_keys = [
        "geometry_id",
        "protocol",
        "split",
        "degree",
        "task",
        "config_name",
    ]
    data = errors.merge(
        features,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    if data[list(VARIABLES)].isna().any().any():
        raise ValueError("Missing explanatory variables after merge.")
    expected_rows = len(MODELS) * len(PROTOCOLS) * len(DEGREES) * 1600
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} analysis rows, got {len(data)}")

    regressions = run_regressions(data)
    write_tables(data, regressions, output)
    write_method_document(output, regressions)

    if not args.skip_plots:
        pooled = regressions[regressions["scope"] == "all"].copy()
        for (model, protocol, degree), task_frame in data.groupby(
            ["model", "protocol", "degree"], sort=False
        ):
            task = f"{protocol}-k{degree}"
            model_slug = model.lower().replace("-", "_")
            for variable in VARIABLES:
                regression_row = pooled[
                    (pooled["model"] == model)
                    & (pooled["task"] == task)
                    & (pooled["variable"] == variable)
                ].iloc[0]
                path = (
                    output
                    / "figures"
                    / "scatter"
                    / variable
                    / f"{model_slug}__protocol_{protocol.lower()}__k{degree}.png"
                )
                make_scatter_plot(task_frame, regression_row, variable, path)
        make_heatmaps(pooled, output / "figures")

    qc = {
        "raw_seed_rows": int(len(raw_errors)),
        "analysis_rows": int(len(data)),
        "regression_rows": int(len(regressions)),
        "pooled_regression_rows": int((regressions["scope"] == "all").sum()),
        "scatter_figures_expected": len(MODELS)
        * len(PROTOCOLS)
        * len(DEGREES)
        * len(VARIABLES),
        "finite_E_final": bool(np.isfinite(data["E_final"]).all()),
        "positive_E_final": bool((data["E_final"] > 0).all()),
        "seed_count": 3,
    }
    with (output / "supporting_data/qc_summary.json").open("w", encoding="utf-8") as file:
        json.dump(qc, file, indent=2)
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
