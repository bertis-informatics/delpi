import numpy as np
import numpy.typing as npt
import numba as nb


@nb.njit(nogil=True, fastmath=True, cache=True)
def weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    weighted_sum = 0.0
    for i in nb.prange(values.shape[0]):
        weighted_sum += values[i] * weights[i]
    return weighted_sum / np.sum(weights)


@nb.njit(nogil=True, fastmath=True, cache=True)
def extract_upper_triangle(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    size = (n * (n - 1)) // 2  # number of elements above diagonal
    result = np.empty(size, dtype=arr.dtype)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            result[idx] = arr[i, j]
            idx += 1

    return result


@nb.njit(nogil=True, fastmath=True, cache=True)
def count_corrcoef(corrcoeff: np.ndarray, cutoff: float) -> int:
    n = corrcoeff.shape[0]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if corrcoeff[i, j] > cutoff:
                count += 1
    return count


@nb.njit(nogil=True, fastmath=True, cache=True)
def pearsonr(x: np.ndarray, y: np.ndarray) -> float:

    assert len(x) > 0
    assert x.ndim == 1
    assert x.shape == y.shape

    x_bar = np.mean(x)
    y_bar = np.mean(y)

    x_centered = x - x_bar
    y_centered = y - y_bar

    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

    return numerator / (denominator + 1e-12)


@nb.njit(nogil=True, fastmath=True, cache=True)
def rowwise_pearsonr(X, y):
    n_samples, n_features = X.shape
    y_mean = np.mean(y)
    y_std = np.std(y)
    corrs = np.empty(n_samples, dtype=np.float32)

    for i in nb.prange(n_samples):
        xi = X[i]
        xi_mean = np.mean(xi)
        xi_std = np.std(xi)

        if xi_std == 0 or y_std == 0:
            corrs[i] = 0.0  # avoid division by zero
        else:
            cov = np.sum((xi - xi_mean) * (y - y_mean))
            corrs[i] = cov / (n_features * xi_std * y_std)

    return corrs


@nb.njit(nogil=True, fastmath=True, cache=True)
def columnwise_pearsonr(X, y):
    n_features, n_samples = X.shape
    y_mean = np.mean(y)
    y_std = np.std(y)
    corrs = np.empty(n_samples)

    for i in nb.prange(n_samples):
        xi = X[:, i]
        xi_mean = np.mean(xi)
        xi_std = np.std(xi)

        if xi_std == 0 or y_std == 0:
            corrs[i] = 0.0
        else:
            cov = 0.0
            for j in range(n_features):
                cov += (xi[j] - xi_mean) * (y[j] - y_mean)
            corrs[i] = cov / (n_features * xi_std * y_std)
    return corrs


@nb.njit(nogil=True, fastmath=True, cache=True)
def calc_xic_correlation(xic_array: np.ndarray) -> np.ndarray:

    num_xics, n_data_points = xic_array.shape

    # (n_fragments, 1)
    profile_mean = np.reshape(
        np.sum(xic_array, axis=1) / n_data_points,
        (num_xics, 1),
    )

    # (n_fragments, n_data_points)
    profile_centered = xic_array - profile_mean

    # (n_fragments, 1)
    profile_std = np.reshape(
        np.sqrt(np.sum(profile_centered**2, axis=1) / n_data_points),
        (num_xics, 1),
    )

    # (n_fragments, n_fragments)
    covariance_matrix = np.dot(profile_centered, profile_centered.T) / n_data_points

    # (n_fragments, n_fragments)
    std_matrix = np.dot(profile_std, profile_std.T)

    # (n_fragments, n_fragments)
    correlation_matrix = covariance_matrix / (std_matrix + 1e-12)
    # output = correlation_matrix

    return correlation_matrix


@nb.njit(nogil=True, fastmath=True, cache=True)
def corrcoef(input_arr):
    n, k = input_arr.shape
    a = np.empty_like(input_arr)

    for i in nb.prange(n):
        if np.any(input_arr[i] != 0):
            mu = input_arr[i].mean()
            sigma = max(input_arr[i].std(), 1e-12)
            a[i] = (input_arr[i] - mu) / sigma
        else:
            a[i] = 0

    ak = a / k
    out = np.empty((n, n))

    for i in nb.prange(n):
        out[i, i] = 1.0
        for j in nb.prange(i + 1, n):
            out[i, j] = ak[i] @ a[j]
            out[j, i] = out[i, j]

    return out


@nb.njit(fastmath=True, nogil=True, cache=True)
def cosine_distance(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        return 1.0

    cosine_sim = dot_product / (norm_u * norm_v)
    cosine_dist = 1.0 - cosine_sim
    return cosine_dist


@nb.njit(fastmath=True, nogil=True, cache=True)
def cosine_similarity_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    N, M = a.shape
    result = np.empty(M, dtype=np.float64)
    b_norm = np.sqrt(np.sum(b**2))
    if b_norm == 0:
        b_norm = 1.0
    b_normalized = b / b_norm

    for j in nb.prange(M):
        col = a[:, j]
        col_norm = np.sqrt(np.sum(col**2))
        if col_norm == 0:
            col_norm = 1.0
        dot = 0.0
        for i in range(N):
            dot += col[i] * b_normalized[i]
        result[j] = dot / col_norm

    return result


@nb.njit
def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    half = 0.5 * np.sum(ws)
    c = 0.0
    for i in range(ws.size):
        c += ws[i]
        if c >= half:
            return xs[i]
    return xs[-1]


def perf_test():
    import time
    from scipy.spatial.distance import cosine

    a = np.random.rand(10000, 10)
    b = np.random.rand(10000, 10)
    st_t = time.perf_counter()
    for u, v in zip(a, b):
        _ = cosine_distance(u, v)
        # _ = pearsonr(u, v)
        # _ = cosine(u, v)
    time.perf_counter() - st_t

    st_t = time.perf_counter()
    for _ in range(100):
        # corr = corrcoef(a)
        corr = calc_xic_correlation(a)
    time.perf_counter() - st_t
