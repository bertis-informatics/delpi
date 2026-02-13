# DelPi

DelPi is an open-source peptide identification tool for mass spectrometry–based proteomics. It applies a pre-trained Transformer encoder to score candidate peptides from raw MS1/MS2 evidence using an acquisition-agnostic representation, enabling a unified workflow across both DIA and DDA data.

## Key Features
- **Deep representation learning:** Scores candidate peptides using a pre-trained Transformer encoder, without relying on handcrafted features.
- **Acquisition-agnostic:** Supports both DDA and DIA within a unified scoring framework.
- **Library-free search:** Uses internally generated in silico spectral libraries.
- **GPU-accelerated inference:** Designed for practical performance on consumer-grade GPUs via PyTorch/CUDA.
- **Experiment-adaptive workflow:** Employs a two-stage search with experiment-level transfer learning to adapt to instrument and chromatographic conditions.

## System Requirements

**Memory:** ≥ 32 GB RAM

**Compute:**
- **NVIDIA GPU with CUDA support required**
- Supported OS: **Linux, Windows**
- macOS (including Apple Silicon/MPS) and CPU-only execution are not supported


**Memory Considerations:**
- DelPi processes input files **one run at a time**
- Peak memory usage depends on the size of an individual raw/mzML file being processed
- Recommended available memory: **(single run file size + ~16 GB)** to accommodate intermediate data structures, model execution, and OS overhead

**Runtime:**
- For a 25 min DIA gradient (human sample, Astral Orbitrap), DelPi completes peptide identification in approximately 20 minutes on a single NVIDIA RTX 4090 GPU
    

## Supported LC-MS/MS Data

- **Formats:** mzML and Thermo RAW (other vendor formats can be converted to mzML using [ProteoWizard MSConvert](https://proteowizard.sourceforge.io))
- **Acquisition modes:** DIA and DDA
- **Ion mobility (IM) data** (e.g., FAIMS, PASEF) is not currently supported, but will be supported soon

## Installation

> **Windows users:** Please use **PowerShell** (not Command Prompt/cmd) for all installation steps. The install scripts (`.ps1`) require PowerShell to run.

### Step 1: Clone the Repository
```bash
git clone https://github.com/bertis-informatics/delpi.git
cd delpi
```

### Step 2: Set Up Virtual Environment

Create a virtual environment using venv (Option A) or conda (Option B). Package installation in later steps uses pip or [uv](https://github.com/astral-sh/uv).

#### Option A: Using venv
```bash
# Create a virtual environment with Python 3.12
uv venv delpi_env --python 3.12
# If not using uv: python -m venv delpi_env

# Activate the virtual environment
# Windows:
delpi_env\Scripts\activate
# macOS/Linux:
source delpi_env/bin/activate
```

#### Option B: Using conda
```bash
# Create a conda environment with Python 3.12
conda create -n delpi_env python=3.12 -y

# Activate the environment
conda activate delpi_env
```

### Step 3: Install PyTorch

Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to obtain the appropriate installation command for your system.

**Example for CUDA 12.8:**
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# If using pip: pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Step 4: Install pymsio

[pymsio](https://github.com/bertis-informatics/pymsio) is bundled in the `pymsio/` directory. The install script downloads the Thermo RawFileReader DLLs and installs pymsio in one step. For additional details, see the [pymsio README](pymsio/README.md).

**Windows PowerShell:**
```powershell
.\pymsio\install.ps1
```

**Linux:**
```bash
chmod +x pymsio/install.sh
./pymsio/install.sh
```



### Step 5: Install DelPi

```bash
uv pip install .
# If using pip: pip install .
```

### Step 6: Verify Installation

```bash
delpi --help
python -c "import delpi; print('DelPi installed successfully!')"
```

## Quick Test

Verify your DelPi installation using publicly available DIA data from the [Skyline tutorial](https://skyline.ms/tutorials/DIA-QE.zip).

1. **Download and extract the DIA test dataset:**
   ```bash
   wget https://skyline.ms/tutorials/DIA-QE.zip
   unzip DIA-QE.zip
   ```

2. **Configure search parameters:**
   
   Copy the example configuration file:
   ```bash
   cp data/example_param.yaml my_config.yaml
   ```
   
   Edit `my_config.yaml` to specify paths for `input_files`, `fasta_file`, `output_directory`, and `database_directory`.

3. **Run the search:**
   ```bash
   delpi my_config.yaml
   ```

4. **Verify output:**
   
   DelPi generates the following files in your specified `output_directory`:
   
   - **`delpi.log`**: Detailed execution log ([example](/examples/output/delpi.log))
   - **`pmsm_results.<tsv|parquet>`**: Peptide-spectrum matches with q-values (format depends on configuration; [example](/examples/output/pmsm_results.tsv))
   - **`protein_group_maxlfq_results.tsv`**: MaxLFQ protein quantification ([example](/examples/output/protein_group_maxlfq_results.tsv))
   
   Compare your results with the provided examples to verify correct installation.

## Getting Started

### 1. Prepare LC-MS/MS Data

Ensure your data files are in a [supported format](#supported-lcmsms-data). If needed, convert to mzML using [ProteoWizard MSConvert](https://proteowizard.sourceforge.io).


### 2. Configure Search Parameters

Create a YAML configuration file based on the [example template](data/example_param.yaml).

**Required fields:**

| Field | Description |
|-------|-------------|
| *acquisition_method* | Acquisition mode (`DIA` or `DDA`) |
| *input_files* or *input_dir* | Paths to LC–MS/MS data files. If *input_dir* is specified, all mzML files within the directory will be automatically processed. |
| *fasta_file* | Protein database in FASTA format |
| *output_directory* | Directory where search results will be written |
| *database_directory* | Directory for storing internally generated in silico spectral libraries (if libraries generated using the same FASTA file and search options already exist, they will be reused) |

**Optional fields:**

Digestion and modification parameters can be adjusted for your experimental setup. Modification names must follow [PSI-MS controlled vocabulary terms](https://www.unimod.org/fields.html).


### 3. Run the Search

Execute DelPi with your configuration file:

```bash
delpi /path/to/your/config.yaml
```

### 4. Output Files

DelPi generates the following output files:

#### Main results report, `pmsm_results.<tsv|parquet>`

<details>
<summary>Click to expand output fields</summary>

| Field name | Description |
|-----------|-------------|
| *frame_num* | Scan number corresponding to the center of the Peptide–Multi-Spectra Match (PmSM) |
| *run_name* | Name of the LC–MS run |
| *modified_sequence* | Peptide sequence including post-translational modifications |
| *precursor_charge* | Charge state of the precursor ion |
| *sequence_length* | Length of the peptide sequence |
| *is_decoy* | Indicator specifying whether the match originates from a decoy sequence |
| *predicted_rt* | Predicted retention time of the peptide |
| *observed_rt* | Observed retention time of the peptide |
| *score* | Raw PmSM score assigned by the DelPi scoring model |
| *global_precursor_q_value* | Global precursor-level q-value across all runs |
| *global_peptide_q_value* | Global peptide-level q-value across all runs |
| *global_protein_group_q_value* | Global protein group-level q-value across all runs |
| *protein_group* | Protein group inferred according to the parsimony principle (FASTA IDs separated by semicolons) |
| *fasta_id* | FASTA IDs associated with the peptide, separated by semicolons |
| *precursor_q_value* | Run-specific precursor-level q-value |
| *peptide_q_value* | Run-specific peptide-level q-value |
| *protein_group_q_value* | Run-specific protein group-level q-value |
| *ms1_area* | Integrated area under the precursor ion chromatogram in MS1 spectra |
| *ms2_area* | *(DIA only, optional)* Precursor abundance quantified from fragment-level signals |

</details>

#### Protein-level quantification results report (DIA only, optional) `protein_group_maxlfq_results.<tsv|parquet>`

<details>
<summary>Click to expand output fields</summary>

| Field name | Description |
|-----------|-------------|
| *run_name* | Name of the LC–MS run |
| *protein_group* | Protein group inferred according to the parsimony principle (FASTA IDs separated by semicolons) |
| *abundance* | Protein abundance calculated using the MaxLFQ algorithm (Cox et al., 2014) |

</details>

---

## Citation

If you use DelPi in your research, please cite:

Park, J., Kim, K., Kang, U.-B., & Kim, S. *DelPi Learns Generalizable Peptide–Signal Correspondence for Mass Spectrometry-Based Proteomics.* bioRxiv (2026). https://doi.org/10.64898/2026.01.06.697814

Preprint: https://www.biorxiv.org/content/10.64898/2026.01.06.697814v1

## License

DelPi is freely available under the [MIT License](LICENSE.txt).

## Contact

For questions, bug reports, or feature requests, please contact **Jungkap Park** at jungkap.park@bertis.com

