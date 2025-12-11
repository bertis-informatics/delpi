import polars as pl
import numpy as np

from delpi.search.dia.lfq_utils import _nb_build_L_b


def _maxlfq_one_protein(
    df_p: pl.DataFrame,
    run_col: str,
    peptide_col: str,
    log_col: str = "logI",
) -> tuple[list, np.ndarray]:
    """
    Cox 2014 MaxLFQ

    Parameters
    ----------
    df_p : pl.DataFrame (columns: run_col, peptide_col, log_col)
    run_col : str
    peptide_col : str
    log_col : str
        log-intensity column name (e.g. "logI")

    Returns
    -------
    runs : run identifier list (in fixed order)
    x : np.ndarray
    """
    # 이 protein에서 관측된 run들의 리스트 (반환 순서 유지)
    runs = df_p[run_col].unique().to_list()
    n_runs = len(runs)
    if n_runs == 0:
        return [], np.array([], dtype=float)
    if n_runs == 1:
        # run 하나 뿐이면 ratio를 만들 수 없으므로, 0으로 놓고 나중에 상수 더해도 됨
        return runs, np.zeros(1, dtype=float)

    # 매핑: run -> 0..n_runs-1
    run_to_idx = {r: i for i, r in enumerate(runs)}
    # 매핑: peptide -> 0..n_pep-1
    pep_vals = df_p[peptide_col].to_list()
    pep_to_idx = {}
    pep_idx_list = []
    next_idx = 0
    for v in pep_vals:
        idx = pep_to_idx.get(v)
        if idx is None:
            pep_to_idx[v] = next_idx
            pep_idx_list.append(next_idx)
            next_idx += 1
        else:
            pep_idx_list.append(idx)

    run_idx_arr = np.array(
        [run_to_idx[v] for v in df_p[run_col].to_list()], dtype=np.int64
    )
    pep_idx_arr = np.array(pep_idx_list, dtype=np.int64)
    logI_arr = np.array(df_p[log_col].to_list(), dtype=np.float64)

    # peptide별로 연속되도록 정렬 (numba에서 그룹 경계 스캔)
    order = np.argsort(pep_idx_arr, kind="mergesort")  # stable sort
    pep_idx_sorted = pep_idx_arr[order]
    run_idx_sorted = run_idx_arr[order]
    logI_sorted = logI_arr[order]

    L, b = _nb_build_L_b(n_runs, pep_idx_sorted, run_idx_sorted, logI_sorted)

    # gauge fixing: x[0] = 0
    L[0, :] = 0.0
    L[:, 0] = 0.0
    L[0, 0] = 1.0
    b[0] = 0.0

    # 선형 시스템 풀기 (가능하면 solve, 안 되면 lstsq)
    try:
        x = np.linalg.solve(L, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(L, b, rcond=None)

    return runs, x


def maxlfq(
    df: pl.DataFrame,
    protein_col: str = "protein_group",
    peptide_col: str = "peptide_index",
    run_col: str = "run_index",
    intensity_col: str = "peptide_abundance",  # ms2_area 같은 값
    min_peptides_per_protein: int = 2,
) -> pl.DataFrame:
    """
    Cox et al. 2014 MaxLFQ 알고리즘(쌍별 log-ratio + 최소제곱)을
    polars DataFrame 기반으로 구현한 버전.

    - protein별로 MaxLFQ를 수행해 run별 protein log-abundance를 추정
    - 전체 데이터의 log(intensity) global median을 기준으로 스케일을 맞춰
      최종적으로는 원래 intensity(ms2_area)와 비슷한 범위의 protein intensity를 반환.

    Parameters
    ----------
    df : pl.DataFrame
        Long-format table with protein, peptide, run, intensity.
    protein_col, peptide_col, run_col, intensity_col : str
        Column names.
    min_peptides_per_protein : int
        Require at least this many distinct peptides per protein
        (너무 작은 protein은 MaxLFQ에 부적합하므로 제외하거나 NaN 처리하는 게 좋음).

    Returns
    -------
    pl.DataFrame
        Wide-format protein intensity matrix:
        rows = protein, columns = runs (MaxLFQ-based protein_abundance).
    """
    # 0) intensity > 0, NaN 제거
    df = df.filter(pl.col(intensity_col).is_not_null() & (pl.col(intensity_col) > 0))

    if df.height == 0:
        return pl.DataFrame({protein_col: [], run_col: [], "protein_abundance": []})

    # 1) log-intensity 추가
    df_log = df.with_columns(pl.col(intensity_col).log().alias("logI"))

    # 전체 데이터에서 log-intensity global median (절대 스케일 기준)
    global_log_median = float(df_log.select(pl.col("logI").median()).item())

    # 2) protein 단위로 MaxLFQ 수행
    records: list[tuple] = []

    for prot, df_p in df_log.group_by(protein_col, maintain_order=False):
        # protein 당 peptide 수 필터링
        n_pep = df_p.select(pl.col(peptide_col).n_unique()).item()
        if n_pep < min_peptides_per_protein:
            # peptide가 너무 적으면 MaxLFQ로 안정적인 추정이 어렵다고 보고 스킵하거나
            # 그대로 simple sum/mean을 쓰는 옵션을 나중에 추가할 수 있음.
            continue

        runs, x = _maxlfq_one_protein(
            df_p=df_p,
            run_col=run_col,
            peptide_col=peptide_col,
            log_col="logI",
        )

        if len(runs) == 0:
            continue

        # x는 run별 log-abundance (relative, gauge x[0]=0).
        # protein 내 평균이 0이 되도록 한 번 더 센터링한 뒤,
        # 전체 데이터 global_log_median을 더해서 스케일을 맞춘다.
        x = np.asarray(x, dtype=float)
        x_centered = x - np.nanmean(x)
        log_protein = x_centered + global_log_median
        protein_intensity = np.exp(log_protein)

        for run, val in zip(runs, protein_intensity):
            records.append((prot[0], run, float(val)))

    if not records:
        return pl.DataFrame({protein_col: [], run_col: [], "abundance": []})

    df_prot_long = pl.DataFrame(
        records, schema=[protein_col, run_col, "abundance"], orient="row"
    )

    return df_prot_long

    # 3) protein × run wide matrix로 pivot
    # protein_matrix = (
    #     df_prot_long
    #     .pivot(
    #         values="protein_abundance",
    #         index=protein_col,
    #         on=run_col,
    #     )
    #     .sort(protein_col)
    # )

    # return protein_matrix
