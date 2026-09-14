from matplotlib import pyplot as plt
import numpy as np
import polars as pl

from delpi.model.input import ExpPeakInput
from delpi.model.rt_calibrator import RetentionTimeCalibrator


def plot_pmsm(x_exp: np.ndarray):

    peak_data = {e.value: x_exp[:, e.index].astype(e.dtype) for e in ExpPeakInput}
    peak_df = pl.DataFrame(peak_data).sort("time_index")

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    x = np.arange(9)

    precursor_peak_df = peak_df.filter(pl.col("is_precursor"))
    for grp, sub_df in precursor_peak_df.group_by(
        ["isotope_index", "ms_level"], maintain_order=True
    ):
        line_style = "dotted" if grp[1] > 1 else "solid"
        y = np.zeros(9)
        y[sub_df["time_index"]] = sub_df["ab"]
        axs[0].plot(x, y, linestyle=line_style)
    axs[0].set_title("Precursor XICs")
    fragment_peak_df = peak_df.filter(
        (~pl.col("is_precursor")) & (pl.col("isotope_index") == 0)
    )
    for grp, sub_df in fragment_peak_df.group_by(
        ["cleavage_index", "charge", "is_prefix"], maintain_order=True
    ):
        y = np.zeros(9)
        y[sub_df["time_index"]] = sub_df["ab"]
        axs[1].plot(x, y)
    axs[1].set_title("Fragment XICs")
    return fig, axs


def plot_rt_mapping(
    rt_calibrator: RetentionTimeCalibrator,
    ref_rt: np.ndarray,
    obs_rt: np.ndarray,
):
    # img_file_path = search_config.output_dir / f"{run.name}.RT_mapping.jpg"

    """Plot and save the RT mapping."""
    x_rt = np.arange(0, 1.01, 0.01)
    pred_rt_df = rt_calibrator.predict(x_rt)

    plt.figure()
    plt.scatter(ref_rt, obs_rt, color="gray", marker=".")
    plt.plot(x_rt, pred_rt_df["predicted_rt"], color="r")
    plt.plot(x_rt, pred_rt_df["rt_lb"], color="blue", linestyle=":")
    plt.plot(x_rt, pred_rt_df["rt_ub"], color="purple", linestyle=":")
    plt.title(f"RT mapping with {len(obs_rt)} PmSMs")
    plt.xlabel("Reference RT")
    plt.ylabel("Observed RT [seconds]")
    return plt.gcf()


def plot_rt_bootstrap_diagnostics(
    rt_calibrator,
    anchors_df: pl.DataFrame,
    fit_df: pl.DataFrame,
    diagnostics: dict,
    bin_edges: np.ndarray,
    success: bool,
):
    """Plot anchor support, robust-fit residuals, and RT-bin coverage."""
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    mapping_ax, residual_ax = axes[0]
    coverage_ax, info_ax = axes[1]

    anchor_ref_rt = anchors_df["ref_rt"].to_numpy()
    anchor_obs_rt = anchors_df["observed_rt"].to_numpy()
    if anchors_df.height:
        mapping_ax.scatter(
            anchor_ref_rt,
            anchor_obs_rt,
            s=7,
            color="0.72",
            alpha=0.35,
            linewidths=0,
            label=f"qualified anchors ({anchors_df.height:,})",
        )

    fit_ref_rt = fit_df["ref_rt"].to_numpy()
    fit_obs_rt = fit_df["observed_rt"].to_numpy()
    inlier_mask = diagnostics.get("inlier_mask")
    if inlier_mask is not None and len(inlier_mask) == fit_df.height:
        inlier_mask = np.asarray(inlier_mask, dtype=bool)
        mapping_ax.scatter(
            fit_ref_rt[inlier_mask],
            fit_obs_rt[inlier_mask],
            s=10,
            color="#1769aa",
            alpha=0.55,
            linewidths=0,
            label=f"fit inliers ({int(inlier_mask.sum()):,})",
        )
        mapping_ax.scatter(
            fit_ref_rt[~inlier_mask],
            fit_obs_rt[~inlier_mask],
            s=13,
            color="#c62828",
            alpha=0.7,
            linewidths=0,
            label=f"fit outliers ({int((~inlier_mask).sum()):,})",
        )

    if anchor_ref_rt.size:
        ref_min, ref_max = float(anchor_ref_rt.min()), float(anchor_ref_rt.max())
    elif bin_edges.size:
        ref_min, ref_max = float(bin_edges[0]), float(bin_edges[-1])
    else:
        ref_min, ref_max = 0.0, 1.0
    ref_grid = np.linspace(ref_min, ref_max, 500)
    prediction_df = rt_calibrator.predict(ref_grid)
    predicted_rt = prediction_df["predicted_rt"].to_numpy()
    mapping_ax.plot(ref_grid, predicted_rt, color="#111111", linewidth=2, label="fit")
    mapping_ax.fill_between(
        ref_grid,
        prediction_df["rt_lb"].to_numpy(),
        prediction_df["rt_ub"].to_numpy(),
        color="#f9a825",
        alpha=0.2,
        label="search bounds",
    )
    mapping_ax.set_xlabel("Reference RT")
    mapping_ax.set_ylabel("Observed RT [seconds]")
    mapping_ax.legend(loc="best", fontsize=8)
    mapping_ax.grid(alpha=0.2)

    if fit_df.height:
        fit_prediction = rt_calibrator.predict(fit_ref_rt)["predicted_rt"].to_numpy()
        residual = fit_obs_rt - fit_prediction
        residual_color = (
            np.where(inlier_mask, "#1769aa", "#c62828")
            if inlier_mask is not None and len(inlier_mask) == fit_df.height
            else "0.45"
        )
        residual_ax.scatter(
            fit_prediction,
            residual,
            s=9,
            c=residual_color,
            alpha=0.55,
            linewidths=0,
        )
        half_width = diagnostics.get("half_width")
        if half_width is not None:
            residual_ax.axhspan(-half_width, half_width, color="#f9a825", alpha=0.15)
    residual_ax.axhline(0, color="#111111", linewidth=1)
    residual_ax.set_xlabel("Predicted RT [seconds]")
    residual_ax.set_ylabel("Residual [seconds]")
    residual_ax.grid(alpha=0.2)

    counts, _ = np.histogram(anchor_ref_rt, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = np.diff(bin_edges) * 0.85
    coverage_ax.bar(centers, counts, width=widths, color="#2e7d32", alpha=0.8)
    coverage_ax.set_xlabel("Reference RT")
    coverage_ax.set_ylabel("Qualified anchors")
    coverage_ax.set_title("RT-bin coverage")
    coverage_ax.grid(axis="y", alpha=0.2)

    status = "SUCCESS" if success else "FALLBACK"
    info_lines = [
        f"status: {status}",
        f"reason: {diagnostics.get('reason', 'unknown')}",
        f"qualified anchors: {anchors_df.height:,}",
        f"fit anchors: {fit_df.height:,}",
        f"degree: {diagnostics.get('degree', 'n/a')}",
        f"inliers: {diagnostics.get('n_inliers', 'n/a')}",
        f"required half-width: {diagnostics.get('required_half_width', float('nan')):.1f} s",
        f"used half-width: {diagnostics.get('half_width', float('nan')):.1f} s",
        f"maximum half-width: {diagnostics.get('max_half_width', float('nan')):.1f} s",
    ]
    info_ax.axis("off")
    info_ax.text(0, 1, "\n".join(info_lines), va="top", family="monospace")
    fig.suptitle(f"Initial DIA RT bootstrap calibration: {status}")
    return fig
