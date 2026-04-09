<div align="center">
  <h1>Sequence Aligner</h1>
  <p><em>all-vs-all pairwise sequence alignment command-line tool</em></p>
  
  [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
  ![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)
  ![Status](https://img.shields.io/badge/status-academic-orange)
</div>

## Overview

SequenceAligner is a command-line tool for performing all-vs-all (all-against-all) pairwise sequence alignments on protein or DNA sequences. It leverages low level CPU optimizations like SIMD instructions (AVX/SSE), memory mapping, efficient cache utilization and optionally GPU acceleration through CUDA.

> [!NOTE]
> This software is optimized for datasets with many short sequences in an all-vs-all alignment context. For single pairwise alignments or very long sequences, other tools like [Parasail](https://github.com/jeffdaily/parasail) may be more suitable.

> [!WARNING]
> This software is under active development. Although validated against libraries like Parasail, some edge cases may remain undocumented or untested. Use with caution. For questions or issues, contact me at [jakodrag345@gmail.com](mailto:jakodrag345@gmail.com) or open an issue on GitHub.

<details open>
<summary><strong>Features</strong></summary>

- Multiple alignment algorithms:
  - Needleman-Wunsch (global alignment)
  - Smith-Waterman (local alignment)
  - Gotoh algorithm with affine gap penalties
- Predefined substitution matrices
- GPU acceleration with CUDA
- [Multiple configurable options](#usage)
- Memory-mapped disk storage for large similarity matrices
- HDF5 output format with optional compression

</details>

## Installation

### System Requirements

#### CPU Version
- Any x86-64 (x64, AMD64, Intel 64 or "64-bit") processor and Operating System (Linux or Windows)
- See: https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels to find which architecture your CPU supports
- Disk space for output files and enough RAM for results (varies with dataset size, see [File Formats](#file-formats))

#### CUDA Version
- Same as CPU version, plus:
- NVIDIA GPU and drivers

### Releases

- ~~Download the latest release from [Releases](https://github.com/jakovdev/SequenceAligner/releases/latest)~~

> [!NOTE]
> Will release once it's more usable.

## Building from source

<details>
<summary><strong>Linux</strong></summary>

### Dependencies
- GCC, CMake
- HDF5 library
- CUDA toolkit (optional, for GPU acceleration)

### Building

```bash
./script/build_all.sh
```

</details>

<details>
<summary><strong>Windows</strong></summary>

0. Download the project files using the green "Code" button on GitHub and select "Download ZIP". Extract the ZIP file to a folder of your choice. Take note of its location.

### (RECOMMENDED) Windows MSVC (CUDA support)

1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  - Required components: Runtime, Development

2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
  - Standalone packages: MSVC (latest x64/x86), vcpkg, any Windows 10/11 SDK, cmake
  - In the start menu, search for `x64 Native Tools Command Prompt for VS 2022` and open it

3. Navigate to the folder you downloaded and extracted the project using (example):

```bat
cd C:\Users\John\Downloads\SequenceAligner
```

4. Building from source:

```bat
.\script\build_all.bat
```

</details>

<details>
<summary><strong>Windows GCC</strong></summary>

### Windows MSYS2 GCC (no CUDA support)

1. Install MSYS2 from https://www.msys2.org/
2. Open the MSYS2 UCRT64 terminal
3. Navigate to the folder you downloaded the project using:

```bash
# Example:
cd /c/Users/John/Downloads/SequenceAligner
```

Replace the folder path to the location you downloaded the project files.
> - MSYS2 uses `/c/...` instead of `C:\...`

4. Install required tools by running:

```bash
# Update package database and core system packages
pacman -Syu

# Install build tools and HDF5
pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-hdf5
```

5. Build the project using:

```bash
./script/build_all.sh cross
```

While this has faster CPU-only version than MSVC, this version does not support CUDA at all.

</details>

You will find the executable inside the `release` folder once you uncompress it.

## Usage

```bash
# Linux
cd path/to/where/you/uncompressed/release
./seqalign [ARGUMENTS]
```

```bat
# Windows
cd path\to\where\you\uncompressed\release
.\seqalign.exe [ARGUMENTS]
```

<details open>
<summary><strong>Command line arguments</strong></summary>

**Required arguments:**
| Argument | Description |
|--------|-------------|
| `-i, --input FILE` | Input file path: FASTA, DSV (CSV, TSV, etc.) format |
| `-o, --output FILE` | Output file path: HDF5 format |
| `-m, --matrix MATRIX` | Scoring matrix (use --list-matrices to see all available matrices) |
| `-a, --align METHOD` | Alignment method: nw, ga, sw |
| `-p, --gap-penalty N` | Linear gap penalty |
| `-s, --gap-open N` | Affine gap open penalty |
| `-e, --gap-extend N` | Affine gap extend penalty |

**Optional arguments:**
| Argument | Description |
|--------|-------------|
| `-l, --list-matrices` | List available substitution matrices |
| `-f, --filter FLOAT` | Filter sequences with similarity above threshold [0.0-1.0] |
| `-z, --compression N` | Compression level for HDF5 datasets [0-9] (default: 0, no compression) |
| `-B, --benchmark` | Enable timing of various steps |
| `-T, --threads N` | Number of threads (default: 0, auto) |
| `-C, --no-cuda` | Disable CUDA |
| `-W, --no-write` | Disable writing to output file |
| `-P, --no-progress` | Disable progress bars |
| `-D, --no-detail` | Disable detailed printing |
| `-F, --force-proceed` | Force proceed without user prompts (for CI) |
| `-Q, --quiet` | Suppress all non-error printing |
| `-V, --verbose` | Enable verbose printing |
| `-h, --help` | Display this help message |

</details>

### Examples

Below are example commands to run the program. Adjust as needed.

> [!NOTE]
> - File paths should be relative to your current directory, not the binary location
> - On Windows use `.\seqalign.exe` instead of `./seqalign` and change `/` to `\` in file paths

```bash
# Run with all required parameters
./seqalign -i datasets/avppred.csv -m blosum50 -a nw -p 4 -o results/avppred.h5

# Using Smith-Waterman algorithm with 8 threads and CUDA disabled
./seqalign -i datasets/avppred.csv -m blosum62 -a sw -s 10 -e 1 -o results/avppred.h5 -T 8 -C

# Gotoh algorithm with affine gaps
./seqalign -i datasets/avppred.csv -m pam250 -a ga -s 12 -e 2 -o results/avppred.h5

# Enable benchmarking mode with verbose output and without creating the HDF5 result
./seqalign -i datasets/avppred.csv -m blosum62 -a nw -p 4 -BVW

# List all available substitution matrices
./seqalign --list-matrices

# List all arguments
./seqalign --help
```

## Algorithm Implementations

- **Needleman-Wunsch**: Global alignment with linear gap penalties
- **Smith-Waterman**: Local alignment with affine gap penalties 
- **Gotoh Algorithm**: Global alignment with affine gap penalties

### Parasail python equivalents
- parasail.nw() is the Gotoh algorithm with affine gaps in SequenceAligner
- To get actual linear gaps in Parasail you need to set the `open` and `extend` parameters to the same value
- This also applies to the Gotoh algorithm in SequenceAligner, but you should use NW since it is faster and takes one value

| Algorithm | Parasail | SequenceAligner |
|---|---|---|
| Needleman-Wunsch | `parasail.nw(..., open=gap, extend=gap, ...)` | `-a nw -p gap` |
| Smith-Waterman | `parasail.sw(..., open=gap_open, extend=gap_extend, ...)` | `-a sw -s gap_open -e gap_extend` |
| Gotoh (affine gaps) | `parasail.nw(..., open=gap_open, extend=gap_extend, ...)` | `-a ga -s gap_open -e gap_extend` |

## File Formats

### Input Format
- **File Type**: FASTA or DSV (CSV, TSV, etc.) format
- **Sequences**:
  - For protein sequences: IUPAC amino acid single letter codes (ARNDCQEGHILKMFPSTWYVBZX*)
  - For nucleotide sequences: IUPAC nucleotide single letter codes (ATGCSWRYKMBVHDN*)
  - At least 2 sequences, each with minimum length of 1 character
- **DSV Specific Requirements**:
  - Must contain one column with biological sequence data (amino acids or nucleotides)
  - Does NOT support quoted fields or escaped delimiters within fields
  - Columns can be separated by commas, tabs, or other delimiters with auto-detection
  - No specific header requirements - the program will scan and identify the sequence column
  - The program will prompt you to select the correct delimiter or header if it can't find one automatically

### Output Format
- **File Type**: HDF5 (.h5) - a common scientific data format
- **Content**:
  - Sequences used during alignment (original or filtered)
  - Similarity matrix where each cell represents the alignment score between sequence pairs
  - Similarity matrix checksum for data integrity verification
- **Size Considerations**:
  - The matrix grows with the square of the sequence count (1,000 sequences = 1 million cells = 4 MB)
    > Each score number (cell) is 4 bytes
  - For very large datasets, the program will automatically use disk-based storage when needed, so check if you have enough free disk storage if aligning a dataset with hundreds of thousands of sequences (100+ GB simlarity matrix!)
  - Compression can be enabled with the `-z` option (0-9 levels) to reduce file size at the cost of processing time
- **Viewing Results**:
  - HDF5 files can be viewed with tools like [HDFView](https://www.hdfgroup.org/downloads/hdfview/) or [myHDF5](https://myhdf5.hdfgroup.org/)
  - Many programming languages have libraries to read HDF5 (Python: h5py, R: rhdf5)

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.

**Academic Use**: This software is intended for academic purposes only. If you use this software in your research, please cite: [CITATION.cff](CITATION.cff).
