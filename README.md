<div align="center">
  <h1>Sequence Aligner</h1>
  <p><em>High performance pairwise sequence alignment tool</em></p>
  
  [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
  ![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)
  ![Status](https://img.shields.io/badge/status-academic-orange)
</div>

## Overview

SequenceAligner is a highly optimized tool for performing rapid pairwise sequence alignments on protein or DNA sequences. It leverages low level optimizations like SIMD instructions (AVX/SSE), memory mapping, and efficient cache utilization to achieve better performance.

<details open>
<summary><strong>Features</strong></summary>

- Multiple alignment algorithms:
  - Needleman-Wunsch (global alignment)
  - Smith-Waterman (local alignment)
  - Gotoh algorithm with affine gap penalties
- [Multiple configurable options](#usage)
- Predefined scoring matrices
- HDF5 output format with compression

</details>

## Installation

### Releases

- Download the latest release from [Releases](https://github.com/jakovdev/SequenceAligner/releases/latest)


### Building from source

<details>
<summary><strong>Linux</strong></summary>

#### Dependencies
- GCC with C99 support
- GNU Make
- HDF5 library

```bash
# Debian/Ubuntu
sudo apt install build-essential libhdf5-dev

# Arch Linux
sudo pacman -S gcc make hdf5
```

#### Building

```bash
# Clone the repository
git clone https://github.com/user/SequenceAligner.git
cd SequenceAligner

# Build the project
make

# Profiles
make help
```
</details>

<details>
<summary><strong>Windows</strong></summary>

#### Prerequisite

1. Install MSYS2 from https://www.msys2.org/
2. Open the MSYS2 UCRT64 terminal
3. Navigate to the folder you downloaded the project using:

```bash
cd /c/Users/John/Downloads/SequenceAligner
```

> - Replace the folder path to the location you downloaded the project files
> - MSYS2 uses `/c/...` instead of `C:\...`

4. Install required tools by running:
```bash
./scripts/msys2_setup.sh
```

#### Building

```bash
# Build the program
mingw32-make
```

```bash
# All available commands
mingw32-make help
```

</details>

## Usage

```bash
# Linux
./bin/seqalign [ARGUMENTS]

# Windows
./bin/seqalign.exe [ARGUMENTS]
```

<details open>
<summary><strong>Command line arguments</strong></summary>

**Required arguments:**
| Argument | Description |
|--------|-------------|
| `-i, --input FILE` | Input CSV file path |
| `-t, --type TYPE` | Sequence type: amino (protein), nucleotide (DNA/RNA) |
| `-a, --align METHOD` | Alignment method: nw, ga, sw |
| `-m, --matrix MATRIX` | Scoring matrix (use --list-matrices to see options) |
| `-p, --gap-penalty N` | Linear gap penalty (required for Needleman-Wunsch) |
| `-s, --gap-start N` | Affine gap start penalty (required for affine gap methods) |
| `-e, --gap-extend N` | Affine gap extend penalty (required for affine gap methods) |

**Optional arguments:**
| Argument | Description |
|--------|-------------|
| `-o, --output FILE` | Output HDF5 file path (required if writing results) |
| `-T, --threads N` | Number of threads (0 = auto) [default: auto] |
| `-z, --compression N` | HDF5 compression level (0-9) [default: 0 (no compression)] |
| `-f, --filter THRESHOLD` | Filter sequences with similarity above threshold |
| `-B, --benchmark` | Enable benchmarking mode |
| `-W, --no-write` | Disable writing to output file |
| `-v, --verbose` | Enable verbose output |
| `-q, --quiet` | Suppress all non-error output |
| `-l, --list-matrices` | List all available scoring matrices |
| `-h, --help` | Display help message |

</details>

### Examples

> [!NOTE]
> - The input file is my own testing dataset provided with the codebase
> - This means all arguments are ones which work for that dataset (like sequence type)
> - You should change the arguments to match your dataset
> - Also, for relative file paths they should be relative to your current directory, not the binary location

```bash
# Run with all required parameters
./bin/seqalign -i datasets/avppred.csv -o results/avppred.h5 -t amino -a nw -m blosum50 -p 4

# Using Smith-Waterman algorithm (requires affine gap parameters) with 8 threads
./bin/seqalign -i datasets/avppred.csv -o results/avppred.h5 -t amino -a sw -m blosum62 -s 10 -e 1 -T 8

# Gotoh algorithm with affine gaps
./bin/seqalign -i datasets/avppred.csv -o results/avppred.h5 -t amino -a ga -m pam250 -s 12 -e 2

# Enable benchmarking mode with verbose output and without creating the HDF5 result
./bin/seqalign -i datasets/avppred.csv -t amino -a nw -m blosum62 -p 4 -B -v

# List all available scoring matrices
./bin/seqalign --list-matrices

# List all arguments
./bin/seqalign --help
```

## Performance Benchmarks

<table>
  <tr>
    <th colspan="3">Test Environment</th>
  </tr>
  <tr>
    <td><a href="/datasets/avppred.csv">Dataset</a></td>
    <td colspan="2">1042 sequences (21.58 avg length)</td>
  </tr>
  <tr>
    <td>Alignments</td>
    <td colspan="2">542,361 pairwise comparisons</td>
  </tr>
  <tr>
    <td>Algorithm</td>
    <td colspan="2">Needleman-Wunsch</td>
  </tr>
  <tr>
    <th>System</th>
    <th>Threads</th>
    <th>Alignment Time</th>
  </tr>
  <tr>
    <td rowspan="2">Linux (Arch)</td>
    <td>16</td>
    <td><strong>0.026s</strong></td>
  </tr>
  <tr>
    <td>1</td>
    <td>0.15s</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="4">Large Dataset Performance</th>
  </tr>
  <tr>
    <td><a href="/datasets/drosophila.csv">Dataset</a></td>
    <td colspan="3">58,746 sequences (17.93 avg length)</td>
  </tr>
  <tr>
    <td>Alignments</td>
    <td colspan="3">1,725,516,885 pairwise comparisons</td>
  </tr>
  <tr>
    <td>Algorithm</td>
    <td colspan="3">Needleman-Wunsch</td>
  </tr>
  <tr>
    <th>System</th>
    <th>Threads</th>
    <th>Alignment Time</th>
    <th>I/O Time</th>
  </tr>
  <tr>
    <td>Linux (Arch)</td>
    <td>16</td>
    <td><strong>53.651s (32 million/s)</strong></td>
    <td>31.746s (hdf5 conversion)</td>
  </tr>
</table>

> [!NOTE]
> - Smith-Waterman and Gotoh algorithms are slower than Needleman-Wunsch
> - Longer sequences require more time to align
> - For very large datasets (exceeding available RAM), the final step of saving results to HDF5 format may become the most time-consuming part of the process

## Implementation Details

<details>
<summary><strong>Algorithm Implementations</strong></summary>

- **Needleman-Wunsch**: Global alignment with linear gap penalties
- **Smith-Waterman**: Local alignment with affine gap penalties 
- **Gotoh Algorithm**: Global alignment with affine gap penalties

All implementations use dynamic programming with optimized matrix operations.
</details>

<details>
<summary><strong>Optimization Techniques</strong></summary>

- SIMD vectorization using AVX/SSE instructions
- Cache friendly memory access patterns
- Memory prefetching
- Thread work stealing for load balancing
- Huge pages for large memory allocations
- Efficient matrix allocation with stack fallback for small sequences
- Memory mapped input file reading and storage for large matrices
</details>

## File Formats

### Input Format
- **File Type**: Simple CSV (comma-separated values) text file
- **Content Requirements**:
  - Must contain one column with biological sequence data (amino acids or nucleotides)
  - No specific header requirements - the program will scan and identify the sequence column
  - The program will prompt you to select the correct one if it can't find one automatically
- **Sequence Format**:
  - For protein sequences: standard one-letter amino acid codes (ACDEFGHIKLMNPQRSTVWY)
  - For nucleotide sequences: standard DNA/RNA bases (ACGT)

### Output Format
- **File Type**: HDF5 (.h5) - a common scientific data format
- **Content**:
  - Contains a similarity matrix where each cell represents the alignment score between sequence pairs
  - Also stores the original or filtered sequences and their lengths
- **Size Considerations**:
  - The matrix grows with the square of the sequence count (1,000 sequences = 1 million cells = 4 MB)
    > Each score number (cell) is 4 bytes
  - For very large datasets, the program will automatically use disk-based storage when needed, so check if you have enough free disk storage if aligning a dataset with hundreds of thousands of sequences (50 to 100+ GB)
- **Viewing Results**:
  - HDF5 files can be viewed with tools like [HDFView](https://www.hdfgroup.org/downloads/hdfview/) or [myHDF5](https://myhdf5.hdfgroup.org/)
  - Many programming languages have libraries to read HDF5 (Python: h5py, R: rhdf5)


## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.

**Academic Use**: This software is intended for academic purposes only. If you use this software in your research, please cite:

```bibtex
@software{SequenceAligner,
  author = {Jakov Dragičević},
  title = {Sequence Aligner},
  year = {2025},
  url = {https://github.com/jakovdev/SequenceAligner}
}
```