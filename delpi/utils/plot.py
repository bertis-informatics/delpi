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
