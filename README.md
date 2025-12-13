# DelPi

DelPi is an open-source peptide identification tool for mass spectrometry–based proteomics. It applies a pre-trained Transformer encoder to score candidate peptides from raw MS1/MS2 evidence using an acquisition-agnostic representation, enabling a unified workflow across both DIA and DDA data.

## Key Features
- **Deep representation learning:** Scores candidate peptides using a pre-trained Transformer encoder, without relying on handcrafted features.
- **Acquisition-agnostic:** Supports both DDA and DIA within a unified scoring framework.
- **Library-free search:** Uses internally generated in silico spectral libraries.
- **GPU-accelerated inference:** Designed for practical performance on consumer-grade GPUs via PyTorch/CUDA.
- **Experiment-adaptive workflow:** Employs a two-stage search with experiment-level transfer learning to adapt to instrument and chromatographic conditions.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv), a fast Python package manager. If you prefer pip, simply replace `uv` commands with `pip` equivalents noted below.

### Step 1: Create a Virtual Environment

```bash
# Navigate to the DelPi directory
cd /path/to/delpi

# Create a virtual environment with Python 3.12
uv venv delpi_env --python 3.12
# If using pip: python -m venv delpi_env

# Activate the virtual environment
# Windows:
delpi_env\Scripts\activate
# macOS/Linux:
source delpi_env/bin/activate
```

### Step 2: Install PyTorch

Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to obtain the appropriate installation command for your system.

**Example for CUDA 12.8:**
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# If using pip: pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install DelPi

```bash
uv pip install -e .
# If using pip: pip install -e .
```

### Step 4: Verify Installation

```bash
delpi --help
python -c "import delpi; print('DelPi installed successfully!')"
```

## Getting Started

### 1. Prepare LC-MS/MS Data

DelPi requires LC-MS/MS data files in **mzML format** (DIA or DDA mode). Vendor raw files can be converted using [ProteoWizard MSConvert](https://proteowizard.sourceforge.io). Support for vendor-specific raw formats will be added in future releases.


### 2. Configure Search Parameters

Create a YAML parameter file. See the [example configuration](data/example_param.yaml) for reference.

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

DelPi generates a tab-separated output file (`pmsm_results.tsv`):

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

For questions, bug reports, feature requests, or suggestions, please contact:

**Jungkap Park**  
Email: jungkap.park@bertis.com
