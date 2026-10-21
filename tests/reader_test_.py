from pymsio import MzmlFileReader


# reader = MzmlFileReader(r'D:\MassSpecData\DDA\2018-NCI7\mzML\06_CPTAC_TMTS1-NCI7_P_JHUZ_20170509_LUMOS.mzML')

reader = MzmlFileReader(
    r"D:\MassSpecData\DIA_Public\2023-HEK293\mzML\20230320_OLEP08_1000ngHeK_uPAC_180k-30min_MontBlanc_2p5ms_0p5_2Th_03.mzML"
)
reader._num_spectra_resolved

reader.num_spectra
