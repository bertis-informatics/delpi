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
    

## Installation

We recommend using [uv](https://github.com/astral-sh/uv), a fast Python package manager. If you prefer pip, simply replace `uv` commands with `pip` equivalents noted below.

### Step 1: Clone the Repository
```bash
git clone https://github.com/bertis-informatics/delpi.git
```

### Step 2: Set Up Virtual Environment
```bash
# Navigate to the DelPi directory
cd delpi

# Create a virtual environment with Python 3.12
uv venv delpi_env --python 3.12
# If using pip: python -m venv delpi_env

# Activate the virtual environment
# Windows:
delpi_env\Scripts\activate
# macOS/Linux:
source delpi_env/bin/activate
```

### Step 3: Install PyTorch

Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to obtain the appropriate installation command for your system.

**Example for CUDA 12.8:**
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# If using pip: pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Step 4: Install DelPi

```bash
uv pip install .
# If using pip: pip install .
```

### (Optional) Provide the Thermo DLLs (only needed for Thermo RAW)

- **Linux only**: ensure Mono is installed (required by pythonnet). Use the helper script:

  ```bash
  ./install_mono.sh
  ```

1. Download (or `git clone`) RawFileReader: https://github.com/thermofisherlsms/RawFileReader
2. Copy the two DLLs from `RawFileReader/Libs/Net471/`:
   - `ThermoFisher.CommonCore.Data.dll`
   - `ThermoFisher.CommonCore.RawFileReader.dll`
3. Make the DLLs discoverable:
   - **Set up an environment variable** `PYMSIO_THERMO_DLL_DIR`
     - Windows example:
       ```powershell
       setx PYMSIO_THERMO_DLL_DIR "<path-to-your-dll-folder>"
       ```
     - Linux example:
       ```bash
       export PYMSIO_THERMO_DLL_DIR="<path-to-your-dll-folder>"
       ```
       *(Add the export line to `~/.bashrc` to keep it persistent.)*
     - Copy the DLLs into the folder referenced by the variable.


### Step 5: Verify Installation

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
   - **`pmsm_results.tsv`**: Peptide-spectrum matches with q-values ([example](/examples/output/pmsm_results.tsv))
   - **`protein_group_maxlfq_results.tsv`**: MaxLFQ protein quantification ([example](/examples/output/protein_group_maxlfq_results.tsv))
   
   Compare your results with the provided examples to verify correct installation.

## Getting Started

### 1. Prepare LC-MS/MS Data

DelPi requires LC-MS/MS data in **mzML format** (DIA or DDA mode). Convert vendor raw files using [ProteoWizard MSConvert](https://proteowizard.sourceforge.io). Native vendor format support will be added in future releases.


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

DelPi generates tab-separated output files including `pmsm_results.tsv`:

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
| *protein_group* | Protein group inferred according to the parsimony principle |
| *precursor_q_value* | Run-specific precursor-level q-value |
| *peptide_q_value* | Run-specific peptide-level q-value |
| *protein_group_q_value* | Run-specific protein group-level q-value |
| *ms1_area* | Integrated area under the precursor ion chromatogram in MS1 spectra |
| *ms2_area* | *(DIA only, optional)* Precursor abundance quantified from fragment-level signals |

</details>

---

## Citation

If you use DelPi in your research, please cite:

*Citation information will be updated upon publication.*

## License

DelPi is freely available under the [MIT License](LICENSE.txt).

## Contact

For questions, bug reports, or feature requests, please contact **Jungkap Park** at jungkap.park@bertis.com

