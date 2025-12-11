# DelPi
DelPi: Deep Learning-based Peptide Identification Search Engine

## Requirements
- Python 3.12+
- PyTorch 2.0+ (automatically installed)
- CUDA 11.8+ (optional, for GPU acceleration)

## Quick Installation with uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that makes installation much faster than traditional pip.

### Step 1: Install uv

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Create Virtual Environment and Install DelPi

```bash
# Navigate to DelPi directory
cd /path/to/delpi

# Create virtual environment with Python 3.12
uv venv delpi_env --python 3.12

# Activate virtual environment
# Windows:
delpi_env\Scripts\activate
# macOS/Linux:
source delpi_env/bin/activate
```

#### Install PyTorch (Choose based on your system)

Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to get the correct installation command for your system.

   ```bash
   # Example commands (use the one from PyTorch website):
   # For CUDA 12.8 on Windows:
   uv pip install torch --index-url https://download.pytorch.org/whl/cu128

   # For CPU only on Linux:
   uv pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

#### Install DelPi
```bash
# Install DelPi in development mode
uv pip install -e .
```

### Step 3: Verify Installation

```bash
# Check command line tool
delpi --help

# Test in Python
python -c "import delpi; print('DelPi installed successfully!')"
```

## Alternative Installation Methods

### Traditional pip installation:
If you prefer using pip without uv:

```bash
# Create virtual environment
python -m venv delpi_env

# Activate environment
# Windows:
delpi_env\Scripts\activate
# macOS/Linux:
source delpi_env/bin/activate

# Install PyTorch (visit https://pytorch.org/get-started/locally/ for correct command)
# Examples:
pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8

# Install DelPi
pip install -e .
```


## Usage

After successful installation: 
### Command Line Interface:
```bash
# Show help
delpi --help

# Run with configuration file
delpi config.yaml
```
```

### Getting Started:
1. [TBU]

## Authors
* Jungkap Park <jungkap.park@bertis.com>

