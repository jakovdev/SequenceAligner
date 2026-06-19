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
- GCC 16, CMake
- HDF5 library (development files)
- CUDA toolkit (optional, for GPU acceleration)

### Building

```bash
./script/build_all.sh
```

You will find the executable inside the `release` folder once you uncompress it.

</details>

<details>
<summary><strong>Windows</strong></summary>

### Dependencies
- [MSYS2](https://www.msys2.org/)
  - Use default install location (`C:\msys64`). If you changed it, adjust the paths in the build instructions accordingly.
- [CUDA Toolkit (optional)](https://developer.nvidia.com/cuda-downloads)
  - CUDA
    - Development
      - Compiler
        - Libraries
          - [x] `CRT`
          - [x] `NVVM`
        - [x] `nvcc`
    - Runtime
      - Libraries
        - [x] `CCCL`
        - [x] `CUDART`
- [Visual Studio Build Tools (optional, for CUDA)](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2026)
  - Individual components
    - [x] `MSVC Build Tools for x64/x86 (Latest)`
    - [x] `Windows Universal CRT SDK`

1. Download the project files using the green "Code" button on GitHub and select "Download ZIP". Extract the ZIP file to a folder of your choice. Take note of its location.

2. Open `x64 Native Tools Command Prompt for VS 2026` if building with CUDA
  - Open `Terminal`/`Powershell` if building without CUDA

3. Navigate to the folder you downloaded the project using:

```cmd
cd "C:\path\to\SequenceAligner"
```

4. Run this command to open an MSYS2 shell with the correct environment variables:

```cmd
C:\msys64\msys2_shell.cmd -ucrt64 -use-full-path -defterm -no-start -here
```

5. Install required tools by running:

```bash
# Update package database and core system packages
pacman -Syu --noconfirm
# If this only updated pacman and/or msys, reopen with the same command as step 4 and run this again. This usually happens if you have an older installation.

# Install build tools and HDF5
pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-tools mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-hdf5
```

6. Build the project using:

```bash
./script/build_all.sh
```

You will find the executable inside the `release` folder once you uncompress it.

</details>

## Usage

### Linux

```bash
cd path/to/where/you/uncompressed/release
./seqalign [ARGUMENTS]
```
### Windows

Double click `seqalign.exe`, it should automatically open a terminal window with instructions.

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
- This also applies to the Gotoh algorithm in SequenceAligner, but in that case you should use NW since it's faster

| Algorithm | Parasail | SequenceAligner |
|---|---|---|
| Needleman-Wunsch | `parasail.nw(..., open=gap, extend=gap, ...)` | `-a nw -p gap` |
| Smith-Waterman | `parasail.sw(..., open=gap_open, extend=gap_extend, ...)` | `-a sw -s gap_open -e gap_extend` |
| Gotoh (affine gaps) | `parasail.nw(..., open=gap_open, extend=gap_extend, ...)` | `-a ga -s gap_open -e gap_extend` |

## File Formats

### Input Format
- **File Type**: FASTA (.fasta, .fas, .fa, etc.) or DSV (.csv, .tsv, etc.) format
- **Sequences**:
  - For protein sequences: IUPAC amino acid single letter codes (ARNDCQEGHILKMFPSTWYVBZX*)
  - For nucleotide sequences: IUPAC nucleotide single letter codes (ATGCSWRYKMBVHDN*)
  - At least 2 sequences, each with minimum length of 1 character
- **DSV Specific Requirements**:
  - Must contain one column with biological sequence data (amino acids or nucleotides)
  - The program will prompt you to select the sequence column if it can't find one automatically

### Output Format
- **File Type**: HDF5 (.h5) format
- **Content**:
  - Sequences used during alignment (original or filtered)
  - Similarity matrix where each cell represents the alignment score between sequence pairs
- **Size Considerations**:
  - Similarity matrix size grows quadratically with sequence count (1000 sequences = 4MB, 50000 sequences = 10GB)
  - Large matrices above RAM limits use temporary disk-based storage, which means you will need to be able to store 2 of them (one temporary and one final)
  - Compression (`-z [0-9]`) reduces the final HDF5 file size but not temporary disk or RAM usage during alignment. Levels above 5 are not recommended due to extreme compression times after alignment is complete.
- **Viewing Results**:
  - HDF5 files can be viewed with tools like [HDFView](https://www.hdfgroup.org/downloads/hdfview/) or [myHDF5](https://myhdf5.hdfgroup.org/)
  - Many programming languages have libraries to read HDF5 (Python: h5py, R: rhdf5)

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.

**Academic Use**: This software is intended for academic purposes only. If you use this software in your research, please cite: [CITATION.cff](CITATION.cff).
