from typing import Union, List, Self
from pathlib import Path
import h5py
import pickle

import numpy as np
import polars as pl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline


class LinearProjectionCalibrator:

    def __init__(
        self,
        min_rt_in_seconds: float,
        max_rt_in_seconds: float,
        rt_tolerance: float = 0.3,
    ):
        assert max_rt_in_seconds > min_rt_in_seconds
        self.min_rt_in_seconds = min_rt_in_seconds
        self.max_rt_in_seconds = max_rt_in_seconds
        self.rt_tolerance = rt_tolerance
        self.min_ref_rt = 0.0
        self.max_ref_rt = 1.0

    @property
    def lc_grad_len(self):
        return self.max_rt_in_seconds - self.min_rt_in_seconds

    def _to_array(self, rt_data: Union[np.ndarray, List, pl.Series]):
        if isinstance(rt_data, pl.Series):
            return rt_data.to_numpy()
        elif isinstance(rt_data, List):
            return np.asarray(rt_data, dtype=np.float32)
        return rt_data

    def _set_lower_upper_bounds(self, df):

        df = (
            df.with_columns(
                pl.col("predicted_rt"),
                (pl.col("predicted_rt") - pl.col("lower")).alias("rt_lb"),
                (pl.col("predicted_rt") + pl.col("upper")).alias("rt_ub"),
            )
            .select(
                pl.all().clip(
                    lower_bound=self.min_rt_in_seconds,
                    upper_bound=self.max_rt_in_seconds,
                )
            )
            .select(pl.col("predicted_rt", "rt_lb", "rt_ub"))
        )
        return df

    def fit(
        self,
        ref_rt: Union[np.ndarray, List, pl.Series],
        *args,
        **kwargs,
    ) -> Self:
        ref_rt_arr = self._to_array(ref_rt)
        self.min_ref_rt = ref_rt_arr.min()
        self.max_ref_rt = ref_rt_arr.max()
        return self

    def predict(self, ref_rt: Union[np.ndarray, List, pl.Series]):

        # "predicted_rt"
        ref_rt_arr = self._to_array(ref_rt)

        conf_interval = self.rt_tolerance * self.lc_grad_len
        scale = self.lc_grad_len / (self.max_ref_rt - self.min_ref_rt)
        predicted_rt = scale * (ref_rt_arr - self.min_ref_rt) + self.min_rt_in_seconds

        df = pl.DataFrame(
            {"predicted_rt": predicted_rt}, schema={"predicted_rt": pl.Float32}
        ).with_columns(
            lower=pl.lit(conf_interval, dtype=pl.Float32),
            upper=pl.lit(conf_interval, dtype=pl.Float32),
        )

        return self._set_lower_upper_bounds(df)


class RetentionTimeCalibrator(LinearProjectionCalibrator):

    def __init__(
        self,
        min_rt_in_seconds: float,
        max_rt_in_seconds: float,
        min_rt_tolerance: float = 0.15,
        max_rt_tolerance: float = 0.25,
        degree: int = 5,
    ):

        super().__init__(min_rt_in_seconds, max_rt_in_seconds)
        assert max_rt_tolerance >= min_rt_tolerance

        self.min_rt_tol_in_seconds = self.lc_grad_len * min_rt_tolerance
        self.max_rt_tol_in_seconds = self.lc_grad_len * max_rt_tolerance
        self.degree = degree
        self.estimator = None

    def fit(
        self,
        ref_rt: Union[np.ndarray, List, pl.Series],
        obs_rt: Union[np.ndarray, List, pl.Series],
    ):
        # Use float64 to make polynomial regression numerically stable across
        # sklearn / LAPACK implementations.
        x_train = self._to_array(ref_rt).astype(np.float64).reshape(-1, 1)
        y_train = self._to_array(obs_rt).astype(np.float64)

        # fitting calibrator model
        estimator = make_pipeline(
            # include_bias=False avoids duplicating the intercept term because
            # LinearRegression already fits an intercept by default.
            PolynomialFeatures(degree=self.degree, include_bias=False),
            LinearRegression(),
        ).fit(x_train, y_train)

        # estimate residuals
        y_pred = estimator.predict(x_train)
        residuals = y_train - y_pred

        #### Residual Modeling with Heteroscedasticity-Aware Error Modeling
        mask = residuals > 0
        upper_residual_estimator = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=False),
            LinearRegression(),
        ).fit(y_pred[mask].reshape((-1, 1)), residuals[mask])

        mask = residuals < 0
        lower_residual_estimator = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=False),
            LinearRegression(),
        ).fit(y_pred[mask].reshape((-1, 1)), np.abs(residuals[mask]))

        self.residual_estimator = (lower_residual_estimator, upper_residual_estimator)

        #### Conformal Prediction with Residual Binning
        # rt_points = np.linspace(y_pred.min() - 1, y_pred.max() + 1, num=20)
        # rt = []
        # rt_scale = []
        # for j in range(1, rt_points.shape[0]):
        #     mask = (y_pred > rt_points[j - 1]) & (y_pred < rt_points[j])
        #     if np.sum(mask) > 32:
        #         rt_point = (rt_points[j - 1] + rt_points[j]) * 0.5
        #         rt.append(rt_point)
        #         q1 = np.quantile(residuals[mask], q=0.05)
        #         q2 = np.quantile(residuals[mask], q=0.5)
        #         q3 = np.quantile(residuals[mask], q=0.95)
        #         rt_scale.append((0.5 * (q2 - q1), 0.5 * (q3 - q2)))

        # self._rt = np.asarray(rt, dtype=np.float32)
        # self._rt_scale = np.asarray(rt_scale, dtype=np.float32)
        self.estimator = estimator

        return self

    def get_confident_interval(self, rt_pred, confidence_interval_sigma):

        #### Residual Modeling with Heteroscedasticity-Aware Error Modeling
        lower_res = self.residual_estimator[0].predict(rt_pred.reshape((-1, 1)))
        upper_res = self.residual_estimator[1].predict(rt_pred.reshape((-1, 1)))
        lower_interval = lower_res * confidence_interval_sigma
        upper_interval = upper_res * confidence_interval_sigma

        #### Conformal Prediction with Residual Binning
        # lower_interval = (
        #     np.interp(rt_pred, self._rt, self._rt_scale[:, 0])
        #     * confidence_interval_sigma
        # )
        # upper_interval = (
        #     np.interp(rt_pred, self._rt, self._rt_scale[:, 1])
        #     * confidence_interval_sigma
        # )

        lower_interval = np.clip(
            lower_interval,
            a_min=self.min_rt_tol_in_seconds,
            a_max=self.max_rt_tol_in_seconds,
        )
        upper_interval = np.clip(
            upper_interval,
            a_min=self.min_rt_tol_in_seconds,
            a_max=self.max_rt_tol_in_seconds,
        )

        return lower_interval, upper_interval

    def predict(self, ref_rt, confidence_interval_sigma=8.0):

        if self.estimator is None:
            raise NotFittedError()

        x = self._to_array(ref_rt).astype(np.float64).reshape(-1, 1)
        rt_pred = self.estimator.predict(x)
        # rt_scale = self.get_rt_scale(rt_pred)
        lower_interval, upper_interval = self.get_confident_interval(
            rt_pred, confidence_interval_sigma
        )
        df = pl.DataFrame(
            {
                "predicted_rt": rt_pred.astype(np.float32),
                "lower": lower_interval.astype(np.float32),
                "upper": upper_interval.astype(np.float32),
            }
        )

        return self._set_lower_upper_bounds(df)

    @classmethod
    def train(
        cls,
        min_rt_in_seconds: float,
        max_rt_in_seconds: float,
        ref_rt: np.ndarray,
        obs_rt: np.ndarray,
        min_rt_tolerance: float = 0.1,
        max_rt_tolerance: float = 0.15,
        figure_path: Path = None,
        degree: int = 5,
    ) -> Self:
        """Convenience method to train the calibrator."""
        calibrator = cls(
            min_rt_in_seconds,
            max_rt_in_seconds,
            min_rt_tolerance=min_rt_tolerance,
            max_rt_tolerance=max_rt_tolerance,
            degree=degree,
        ).fit(ref_rt, obs_rt)

        if figure_path is not None:
            from delpi.utils.plot import plot_rt_mapping, plt

            fig = plot_rt_mapping(calibrator, ref_rt, obs_rt)
            fig.savefig(figure_path)
            plt.close(fig)

        return calibrator

    def save_to_hdf(self, hdf_path: Path, group_name: str = "rt_calibrator"):

        with h5py.File(hdf_path, "a") as f:
            if group_name in f:
                del f[group_name]

            group = f.create_group(group_name)

            # 기본 파라미터들 저장
            group.attrs["min_rt_in_seconds"] = self.min_rt_in_seconds
            group.attrs["max_rt_in_seconds"] = self.max_rt_in_seconds
            group.attrs["min_rt_tol_in_seconds"] = self.min_rt_tol_in_seconds
            group.attrs["max_rt_tol_in_seconds"] = self.max_rt_tol_in_seconds
            group.attrs["degree"] = self.degree

            # sklearn estimator 저장 (pickle로 직렬화)
            if self.estimator is not None:
                estimator_bytes = pickle.dumps(self.estimator)
                group.create_dataset(
                    "estimator", data=np.frombuffer(estimator_bytes, dtype=np.uint8)
                )

                # residual estimators 저장
                lower_res_bytes = pickle.dumps(self.residual_estimator[0])
                upper_res_bytes = pickle.dumps(self.residual_estimator[1])

                group.create_dataset(
                    "lower_residual_estimator",
                    data=np.frombuffer(lower_res_bytes, dtype=np.uint8),
                )
                group.create_dataset(
                    "upper_residual_estimator",
                    data=np.frombuffer(upper_res_bytes, dtype=np.uint8),
                )
            else:
                group.attrs["estimator_fitted"] = False

    @classmethod
    def load_from_hdf(cls, hdf_path: Path, group_name: str = "rt_calibrator") -> Self:

        with h5py.File(hdf_path, "r") as f:
            if group_name not in f:
                raise KeyError(f"Group '{group_name}' not found in HDF file")

            group = f[group_name]

            min_rt_tolerance = group.attrs["min_rt_tol_in_seconds"] / (
                group.attrs["max_rt_in_seconds"] - group.attrs["min_rt_in_seconds"]
            )
            max_rt_tolerance = group.attrs["max_rt_tol_in_seconds"] / (
                group.attrs["max_rt_in_seconds"] - group.attrs["min_rt_in_seconds"]
            )

            calibrator = cls(
                min_rt_in_seconds=group.attrs["min_rt_in_seconds"],
                max_rt_in_seconds=group.attrs["max_rt_in_seconds"],
                min_rt_tolerance=min_rt_tolerance,
                max_rt_tolerance=max_rt_tolerance,
                degree=group.attrs["degree"],
            )

            # estimator가 저장되어 있으면 로드
            if "estimator" in group:
                estimator_bytes = group["estimator"][()].tobytes()
                calibrator.estimator = pickle.loads(estimator_bytes)

                # residual estimators 로드
                lower_res_bytes = group["lower_residual_estimator"][()].tobytes()
                upper_res_bytes = group["upper_residual_estimator"][()].tobytes()

                lower_estimator = pickle.loads(lower_res_bytes)
                upper_estimator = pickle.loads(upper_res_bytes)
                calibrator.residual_estimator = (lower_estimator, upper_estimator)

            return calibrator

    @staticmethod
    def train_aligner(ref_df, rt_df, rt_column="x_rt", degree=2):

        assert ref_df.shape[0] == ref_df["precursor_index"].n_unique()
        assert rt_df.shape[0] == rt_df["precursor_index"].n_unique()

        df = ref_df.join(rt_df, on="precursor_index", how="inner")

        estimator = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression(),
        ).fit(
            df[f"{rt_column}_right"].to_numpy().reshape((-1, 1)),
            df[rt_column].to_numpy().reshape((-1, 1)),
        )

        return estimator


class BootstrapRTCalibrator(LinearProjectionCalibrator):
    """MAD-robust, degree-<=`max_degree` RT calibrator used only by the DIA
    bootstrap calibration pipeline (:mod:`delpi.search.dia.rt_bootstrap`).
    Not persisted via HDF; used only transiently before the first full DIA
    search.

    Follows the same scikit-learn
    ``Pipeline(PolynomialFeatures, LinearRegression)`` pattern as
    :class:`RetentionTimeCalibrator`, but unlike it, this iteratively
    rejects outliers (MAD clipping), picks the highest degree (up to
    `max_degree`) that still yields a monotonic, positive-trend fit, and
    derives its RT half-width from the residual distribution rather than a
    separately-fit residual model.
    """

    def __init__(
        self,
        min_rt_in_seconds: float,
        max_rt_in_seconds: float,
        max_degree: int = 3,
        mad_clip_thresh: float = 3.5,
        max_mad_iters: int = 3,
        min_rt_tolerance: float = 0.10,
        max_rt_tolerance: float = 0.15,
    ):
        super().__init__(min_rt_in_seconds, max_rt_in_seconds)
        assert max_rt_tolerance >= min_rt_tolerance

        self.max_degree = max_degree
        self.mad_clip_thresh = mad_clip_thresh
        self.max_mad_iters = max_mad_iters
        self.min_rt_tol_in_seconds = self.lc_grad_len * min_rt_tolerance
        self.max_rt_tol_in_seconds = self.lc_grad_len * max_rt_tolerance
        self.degree = None
        self.estimator = None
        self.half_width_in_seconds = None

    def _fit_degree(
        self, x_train: np.ndarray, y_train: np.ndarray, degree: int, scale_floor: float
    ):
        """One MAD-clipping/refitting cycle; stops early once the inlier
        mask stabilizes. Returns ``(estimator_or_None, inlier_mask)``."""
        inlier_mask = np.ones(x_train.shape[0], dtype=bool)
        estimator = None
        for _ in range(self.max_mad_iters):
            if int(np.sum(inlier_mask)) < degree + 1:
                return None, inlier_mask
            estimator = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                LinearRegression(),
            ).fit(x_train[inlier_mask], y_train[inlier_mask])
            residual = y_train - estimator.predict(x_train)
            center = np.median(residual[inlier_mask])
            mad = max(
                1.4826 * np.median(np.abs(residual[inlier_mask] - center)), scale_floor
            )
            new_mask = np.abs(residual - center) <= self.mad_clip_thresh * mad
            if np.array_equal(new_mask, inlier_mask):
                break
            inlier_mask = new_mask
        return estimator, inlier_mask

    def fit(
        self,
        ref_rt: Union[np.ndarray, List, pl.Series],
        obs_rt: Union[np.ndarray, List, pl.Series],
        scale_floor: float = 1.0,
    ) -> tuple:
        """Fit a bootstrap RT mapping. Returns ``(self_or_None, diagnostics)``;
        ``diagnostics`` always has a ``reason`` key and, on success, an
        ``inlier_mask`` aligned with the input ``ref_rt``/``obs_rt`` order.
        """
        x_train = self._to_array(ref_rt).astype(np.float64).reshape(-1, 1)
        y_train = self._to_array(obs_rt).astype(np.float64)
        n = x_train.shape[0]
        if (
            n < 3
            or not np.all(np.isfinite(x_train))
            or not np.all(np.isfinite(y_train))
        ):
            return None, {"reason": "too_few_or_nonfinite_anchors", "n": n}
        if not np.isfinite(self.lc_grad_len) or self.lc_grad_len <= 0:
            return None, {"reason": "degenerate_run_rt_bounds"}

        scale_floor = max(float(scale_floor), 1e-6)
        x_grid = np.linspace(x_train.min(), x_train.max(), 50).reshape(-1, 1)

        estimator, inlier_mask, used_degree = None, np.ones(n, dtype=bool), None
        for degree in range(min(self.max_degree, n - 1), 0, -1):
            candidate, candidate_mask = self._fit_degree(
                x_train, y_train, degree, scale_floor
            )
            if candidate is None:
                continue
            predicted_grid = candidate.predict(x_grid)
            if not np.all(np.isfinite(predicted_grid)) or np.any(
                np.diff(predicted_grid) < -1e-6
            ):
                continue
            if degree == 1 and predicted_grid[-1] <= predicted_grid[0]:
                continue
            estimator, inlier_mask, used_degree = candidate, candidate_mask, degree
            break

        if estimator is None:
            return None, {"reason": "non_monotonic_or_invalid_fit", "n": n}

        residual = y_train - estimator.predict(x_train)
        center = float(np.median(residual[inlier_mask]))
        required_half_width = float(np.quantile(np.abs(residual - center), 0.95))
        half_width = max(required_half_width, self.min_rt_tol_in_seconds)

        diagnostics = {
            "reason": "ok",
            "degree": used_degree,
            "n_anchors": n,
            "n_inliers": int(np.sum(inlier_mask)),
            "required_half_width": required_half_width,
            "half_width": half_width,
            "max_half_width": self.max_rt_tol_in_seconds,
            "inlier_mask": inlier_mask,
        }
        if half_width > self.max_rt_tol_in_seconds:
            diagnostics["reason"] = "residual_too_wide"
            return None, diagnostics

        self.estimator = estimator
        self.degree = used_degree
        self.half_width_in_seconds = half_width
        return self, diagnostics

    def predict(self, ref_rt: Union[np.ndarray, List, pl.Series]) -> pl.DataFrame:
        if self.estimator is None:
            raise NotFittedError()

        x = self._to_array(ref_rt).astype(np.float64).reshape(-1, 1)
        x_flat = x.reshape(-1)
        finite_mask = np.isfinite(x_flat)

        # sklearn's PolynomialFeatures rejects non-finite input outright, so
        # non-finite ref_rt is mapped directly to the RT bounds instead of
        # being extrapolated through the fitted model.
        predicted = np.empty(x_flat.shape[0], dtype=np.float64)
        if np.any(finite_mask):
            predicted[finite_mask] = self.estimator.predict(x[finite_mask])
        predicted[np.isnan(x_flat)] = self.min_rt_in_seconds
        predicted[x_flat == np.inf] = self.max_rt_in_seconds
        predicted[x_flat == -np.inf] = self.min_rt_in_seconds

        df = pl.DataFrame(
            {"predicted_rt": predicted.astype(np.float32)},
            schema={"predicted_rt": pl.Float32},
        ).with_columns(
            lower=pl.lit(self.half_width_in_seconds, dtype=pl.Float32),
            upper=pl.lit(self.half_width_in_seconds, dtype=pl.Float32),
        )
        return self._set_lower_upper_bounds(df)


def test():
    from delpi.model.rt_calibrator import RetentionTimeCalibrator
    import time

    rt_calibrator = RetentionTimeCalibrator.train(
        min_rt_in_seconds=0,
        max_rt_in_seconds=1000,
        ref_rt=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        obs_rt=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    )

    rt_calibrator.save_to_hdf(
        r"D:\MassSpecData\DIA_LIBD\delpi\20240215_Ast_Neo_150uID_4mz_DIA_400-900_10maxIT_250agc_27nce-24m_Control_C1.delpi.h5"
    )

    rt_calib2 = RetentionTimeCalibrator.load_from_hdf(
        r"D:\MassSpecData\DIA_LIBD\delpi\20240215_Ast_Neo_150uID_4mz_DIA_400-900_10maxIT_250agc_27nce-24m_Control_C1.delpi.h5"
    )
