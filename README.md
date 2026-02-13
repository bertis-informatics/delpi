# Pymsio
Pymsio is a lightweight module for reading mass-spectrometry data files into a unified NumPy/Polars representation.  
Its design and implementation are based on the AlphaRaw project: https://github.com/MannLabs/alpharaw/.

It currently supports:

- **Thermo RAW** files (via `pythonnet` + Thermo Fisher CommonCore DLLs)
- **mzML** files

Both formats are exposed through a common interface.

---

## Requirements
- **Supported OS**: Windows, Linux 
- **Python**: **>= 3.8**
- **Thermo RAW**: 
   - Requires Thermo Fisher CommonCore DLLs (`ThermoFisher.CommonCore.Data.dll`, `ThermoFisher.CommonCore.RawFileReader.dll`) obtained from the RawFileReader project (https://github.com/thermofisherlsms/RawFileReader). 
   - Linux also needs Mono (use `install_mono.sh`).

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/bertis-informatics/pymsio.git
   cd pymsio
   ```

2. **Quick Install (recommended)**

   The install script downloads the Thermo RawFileReader DLLs (with license agreement) and installs pymsio in one step.

   **Windows PowerShell**

   ```powershell
   .\install.ps1
   ```

   **Linux / macOS**

   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Manual Install**

   <details>
   <summary>Click to expand manual installation steps</summary>

   #### a. Provide the Thermo DLLs (only needed for Thermo RAW)

   - **Linux only**: ensure Mono is installed (required by pythonnet). Use the helper script:

     ```bash
     ./install_mono.sh
     ```

   1. Download (or `git clone`) RawFileReader: https://github.com/thermofisherlsms/RawFileReader
   2. Copy the two DLLs from `RawFileReader/Libs/Net471/`:
      - `ThermoFisher.CommonCore.Data.dll`
      - `ThermoFisher.CommonCore.RawFileReader.dll`
   3. Make the DLLs discoverable:
      - **Option A — Bundle DLLs inside the package** `<path-to-pymsio>/pymsio/dlls/thermo_fisher/`
        - Copy the DLLs into `pymsio/dlls/thermo_fisher/` *before* running `pip install -e .` so they ship with the installation.
        - Example:
          ```bash
          mkdir -p pymsio/dlls/thermo_fisher
          cp /path/to/RawFileReader/Libs/Net471/*.dll /path/to/pymsio/pymsio/dlls/thermo_fisher/
          ```
      - **Option B — Set up an environment variable** `PYMSIO_THERMO_DLL_DIR`
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

   #### b. Install pymsio

   ```bash
   pip install .
   ```

   </details>

---

## Quick Test

Run the test suite with [pytest](https://docs.pytest.org/):

```bash
# Basic tests (import check, DLL binding check)
pytest tests/

# With reader tests (provide file paths)
pytest tests/ --raw "path/to/file.raw" --mzml "path/to/file.mzML"
```

Tests for readers whose file path is not provided will be **automatically skipped**.

---

## Code Example

<details>
<summary>Click to expand</summary>

```python
from pathlib import Path
from pymsio.readers import ReaderFactory

path = Path("path/to/your/file.raw")  # or .mzML

# 1) Get appropriate reader (Thermo RAW or mzML)
reader = ReaderFactory.get_reader(path)

# 2) Read metadata (Polars DataFrame)
meta_df = reader.get_meta_df()
print(meta_df.head())

# 3) Read one frame (np.ndarray, shape (N, 2), columns: [mz, intensity])
frame_num = int(meta_df.item(0, "frame_num"))
peaks = reader.get_frame(frame_num)
print(peaks.shape)

# 4) Load full dataset
msdata = reader.load()
print(msdata.peak_arr.shape)
```

</details>

---

## License

This project is licensed under the [Apache License 2.0](LICENSE.txt).

> **Note:** The Thermo RawFileReader DLLs are proprietary and subject to their own [license](https://github.com/thermofisherlsms/RawFileReader/blob/main/License.doc). They are **not** included in this repository.
