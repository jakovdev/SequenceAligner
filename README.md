<div align="center">
  <h1>Sequence Aligner</h1>
  <p><em>High performance all-vs-all pairwise sequence alignment tool</em></p>
  
  [![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
  ![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)
  ![Status](https://img.shields.io/badge/status-academic-orange)
</div>

## Overview

SequenceAligner is a highly optimized tool for performing rapid all-vs-all (all-against-all) pairwise sequence alignments on protein or DNA sequences. It leverages low level CPU optimizations like SIMD instructions (AVX/SSE), memory mapping, efficient cache utilization and optionally GPU acceleration through CUDA to achieve better performance.

<details open>
<summary><strong>Features</strong></summary>

- Multiple alignment algorithms:
  - Needleman-Wunsch (global alignment)
  - Smith-Waterman (local alignment)
  - Gotoh algorithm with affine gap penalties
- GPU acceleration with CUDA support
- CPU optimizations:
  - SIMD vectorization (AVX512/AVX2/SSE)
  - Efficient multithreading with minimal overhead
  - Memory-mapped file I/O
  - Sequence memory pools
- [Multiple configurable options](#usage)
- Predefined scoring matrices
- HDF5 output format with optional compression

</details>

## Installation

### Releases

- Download the latest release from [Releases](https://github.com/jakovdev/SequenceAligner/releases/latest)


### Building from source

<details>
<summary><strong>Linux</strong></summary>

#### Dependencies
- GCC with C99 support or later
- GNU Make
- HDF5 library
- CUDA toolkit (optional, for GPU acceleration)

```bash
# Debian/Ubuntu
sudo apt install build-essential libhdf5-dev
# For CUDA support, install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Arch Linux
sudo pacman -S gcc make hdf5
# For CUDA support
sudo pacman -S cuda
```

#### Building

```bash
# Clone the repository
git clone https://github.com/user/SequenceAligner.git
cd SequenceAligner

# Build the project
make

# Build with CUDA support (if available)
make cuda

# Other build profiles
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

# All available commands
mingw32-make help
```

> [!NOTE]
> CUDA support is currently not yet available in the Windows build.

</details>

## Usage

```bash
# Linux (CPU only version)
./bin/seqalign [ARGUMENTS]

# Linux (CUDA+CPU version)
./bin/seqalign-cuda [ARGUMENTS]

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
| `-s, --gap-open N` | Affine gap open penalty (required for affine gap methods) |
| `-e, --gap-extend N` | Affine gap extend penalty (required for affine gap methods) |

**Optional arguments:**
| Argument | Description |
|--------|-------------|
| `-o, --output FILE` | Output HDF5 file path (required if writing results) |
| `-T, --threads N` | Number of threads (0 = auto) [default: auto] |
| `-z, --compression N` | HDF5 compression level (0-9) [default: 0 (no compression)] |
| `-f, --filter THRESHOLD` | Filter sequences with similarity above threshold |
| `-B, --benchmark` | Enable benchmarking mode |
| `-C, --no-cuda` | Disable CUDA |
| `-W, --no-write` | Disable writing to output file |
| `-D, --no-detail` | Disable detailed printing |
| `-v, --verbose` | Enable verbose printing |
| `-q, --quiet` | Suppress all non-error printing |
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

# Run with CUDA
./bin/seqalign-cuda -i datasets/avppred.csv -o results/avppred.h5 -t amino -a nw -m blosum50 -p 4

# Run with CUDA but disable it (equivalent to CPU version)
./bin/seqalign-cuda -i datasets/avppred.csv -o results/avppred.h5 -t amino -a nw -m blosum50 -p 4 -C

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

<div class="benchmarks">
  <table>
    <tr>
      <th colspan="4">Test Environment</th>
    </tr>
    <tr>
      <td><a href="/datasets/avppred.csv">Dataset</a></td>
      <td colspan="3">1042 sequences (21.58 avg length)</td>
    </tr>
    <tr>
      <td>Alignments</td>
      <td colspan="3">542,361 pairwise comparisons</td>
    </tr>
    <tr>
      <td>Algorithm</td>
      <td colspan="3">Needleman-Wunsch</td>
    </tr>
    <tr>
      <th>System</th>
      <th>Hardware</th>
      <th>Threads</th>
      <th>Alignment Time</th>
    </tr>
    <tr>
      <td rowspan="3">Linux (Arch)</td>
      <td>CPU (AMD Ryzen 7 5700G)</td>
      <td>16</td>
      <td><strong>0.026s</strong></td>
    </tr>
    <tr>
      <td>CPU (AMD Ryzen 7 5700G)</td>
      <td>1</td>
      <td>0.15s</td>
    </tr>
    <tr>
      <td>GPU (NVIDIA GTX 1070 Ti)</td>
      <td>1024</td>
      <td><strong>0.028s</strong></td>
    </tr>
  </table>

  <table>
    <tr>
      <th colspan="5">Large Dataset Performance</th>
    </tr>
    <tr>
      <td><a href="/datasets/drosophila.csv">Dataset</a></td>
      <td colspan="4">58,746 sequences (17.93 avg length)</td>
    </tr>
    <tr>
      <td>Alignments</td>
      <td colspan="4">1,725,516,885 pairwise comparisons</td>
    </tr>
    <tr>
      <td>Algorithm</td>
      <td colspan="4">Needleman-Wunsch</td>
    </tr>
    <tr>
      <th>System</th>
      <th>Hardware</th>
      <th>Threads</th>
      <th>Alignment Time</th>
      <th>I/O Time</th>
    </tr>
    <tr>
      <td>Linux (Arch)</td>
      <td>CPU (AMD Ryzen 7 5700G)</td>
      <td>16</td>
      <td>53.651s (32 million/s)</td>
      <td>31.746s</td>
    </tr>
  </table>
</div>

> [!NOTE]
> - Smith-Waterman and Gotoh algorithms are slower than Needleman-Wunsch
> - CUDA acceleration provides speedups for Smith-Waterman, Gotoh and datasets with longer sequences
> - - GA CPU: 0.08s, CUDA: 0.05s for 1042 sequences, avg len: 21.58 @[avppred.csv](datasets/avppred.csv)
> - - NW CPU: 4.049s, CUDA: 3.931s for 9409 sequences, avg len: 30.47 @[amp.csv](datasets/amp.csv)
> - - CPU still faster for datasets with short sequences and/or Needleman-Wunsch algorithm
> - For very large datasets (exceeding available RAM), the alignments will be performed in batches and written to disk before being converted to HDF5 format

## Implementation Details

<details>
<summary><strong>Algorithm Implementations</strong></summary>

- **Needleman-Wunsch**: Global alignment with linear gap penalties
- **Smith-Waterman**: Local alignment with affine gap penalties 
- **Gotoh Algorithm**: Global alignment with affine gap penalties

All implementations use dynamic programming with optimized matrix operations.

> [!NOTE]
> Parasail python equivalents
> - nw in Parasail is actually the Gotoh algorithm with affine gaps
> - To get actual linear gaps you need to set the `open` and `extend` parameters to the same value
> - This also applies to the Gotoh algorithm in this project, but you should use nw since it is faster
> - **Needleman-Wunsch**: `parasail.nw(..., open=gap, extend=gap, ...)`
> - **Smith-Waterman**: `parasail.sw()`
> - **Gotoh Algorithm**: `parasail.nw()`

</details>

<details>
<summary><strong>CPU Optimization Techniques</strong></summary>

- SIMD vectorization using AVX/SSE instructions
- Cache friendly memory access patterns
- Memory prefetching
- Mutex-based thread work allocation with dynamic batch sizing
- Huge pages for large memory allocations
- Efficient matrix allocation with stack fallback for small sequences
- Memory mapped input file reading and storage for large matrices
- Sequence memory pools for fast sequence storage
</details>

<details>
<summary><strong>CUDA GPU Optimization Techniques</strong></summary>

- Constant memory usage for frequently accessed data and smaller datasets
- Device-specific tuning of thread and block dimensions
- Efficient data transfer to and from GPU memory
- Memory-mapped matrix storage for large datasets
- Batched execution for datasets exceeding GPU memory
- Triangular matrix computation to reduce redundant calculations
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

## System Requirements

### CPU Version
- Any x86-64 processor
- Enough RAM to store sequences from a dataset
- Disk space for output files (varies with dataset size)

### CUDA Version
- NVIDIA GPU with compute capability 3.5 or higher
- CUDA toolkit 10.0 or newer
- Appropriate NVIDIA drivers

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