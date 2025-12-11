from matplotlib import pyplot as plt
import numpy as np
import polars as pl

from delpi.model.input import ExpPeakInput
from delpi.model.rt_calibrator import RetentionTimeCalibrator


def plot_pmsm(x_exp: np.ndarray):

    peak_data = {e.value: x_exp[:, e.index].astype(e.dtype) for e in ExpPeakInput}
    peak_df = pl.DataFrame(peak_data).sort("time_index")

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    x = np.arange(9)

    precursor_peak_df = peak_df.filter(pl.col("is_precursor"))
    for grp, sub_df in precursor_peak_df.group_by(
        ["isotope_index", "ms_level"], maintain_order=True
    ):
        line_style = "dotted" if grp[1] > 1 else "solid"
        y = np.zeros(9)
        y[sub_df["time_index"]] = sub_df["ab"]
        axs[0].plot(x, y, linestyle=line_style)

    fragment_peak_df = peak_df.filter(
        (~pl.col("is_precursor")) & (pl.col("isotope_index") == 0)
    )
    for grp, sub_df in fragment_peak_df.group_by(
        ["cleavage_index", "charge", "is_prefix"], maintain_order=True
    ):
        y = np.zeros(9)
        y[sub_df["time_index"]] = sub_df["ab"]
        axs[1].plot(x, y)
        y
    # fig.savefig("./temp/xic.jpg")
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
    plt.title(f"RT mapping with {len(obs_rt)} PMSMs")
    plt.xlabel("Reference RT")
    plt.ylabel("Observed RT [seconds]")
    # plt.savefig(img_file_path)
    return plt.gcf()


# def plot_psm_array(
#     psm_array,
#     ion_type_list,
#     isotope_index=0,
#     disp_neutral_loss=None,
#     disp_idx=1,
#     time_array=None,
# ):
#     # [#cleavages, #ion_types, 3, #iso, #frames]
#     # ion_types are ordered by (charge, base_ion_type)
#     # disp_idx:
#     #       0: ppm error
#     #       1: abundance (intensity)
#     #       2: left-tail prob

#     min_charge = np.min([ion_type.charge for ion_type in ion_type_list])
#     max_charge = np.max([ion_type.charge for ion_type in ion_type_list])
#     num_charges = max_charge - min_charge + 1

#     num_neutral_loss_types = len(
#         set([ion_type.neutral_loss for ion_type in ion_type_list])
#     )
#     num_cols = int(len(ion_type_list) / (num_charges * num_neutral_loss_types))

#     num_cleavages, _, _, _, num_frames = psm_array.shape
#     fig, axs = plt.subplots(num_charges, num_cols, sharex=True, sharey=True)
#     if time_array is None:
#         x = np.arange(num_frames) + 1
#     else:
#         x = time_array
#     X = matlib.repmat(x, num_cleavages, 1)

#     for k, ion_type in enumerate(ion_type_list):
#         if disp_neutral_loss is not None and disp_neutral_loss != ion_type.neutral_loss:
#             continue

#         i = ion_type.charge - min_charge
#         j = int(k / (num_neutral_loss_types)) % num_charges

#         Y = psm_array[:, k, disp_idx, isotope_index, :]
#         _ = axs[i, j].plot(X.T, Y.T)

#         if disp_neutral_loss is None:
#             title_str = f"{ion_type.base_ion_type.symbol} {ion_type.charge}+"
#         else:
#             title_str = f"{ion_type.base_ion_type.symbol}{ion_type.neutral_loss.name} {ion_type.charge}+"

#         if axs[i, j].title.get_text() == "":
#             axs[i, j].set_title(title_str)

#     return fig, axs
