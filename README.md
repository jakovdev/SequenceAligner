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

<details>
<summary><strong>Dependencies</strong></summary>

- GCC with C99 support
- GNU Make
- HDF5 library

### Linux
```bash
# Debian/Ubuntu
sudo apt install build-essential libhdf5-dev

# Arch Linux
sudo pacman -S gcc make hdf5
```

### Windows

- Windows support coming soon

</details>

<details>
<summary><strong>Building from source</strong></summary>

```bash
# Clone the repository
git clone https://github.com/user/SequenceAligner.git
cd SequenceAligner

# Build the project
make

# For Windows cross compilation (not tested yet)
make cross
```
</details>

## Usage

```bash
./bin/main [OPTIONS]
```

<details open>
<summary><strong>Command line options</strong></summary>

| Option | Description |
|--------|-------------|
| `-i, --input FILE` | Input CSV file path [default: ./datasets/avppred.csv] |
| `-o, --output FILE` | Output HDF5 file path [default: ./results/avppred_results.h5] |
| `-a, --align METHOD` | Alignment method: nw, ga, sw [default: nw] |
| `-m, --matrix MATRIX` | Scoring matrix: blosum50, blosum62 [default: blosum50] |
| `-p, --gap-penalty N` | Linear gap penalty for NW [default: 4] |
| `-s, --gap-start N` | Affine gap start penalty for GA/SW [default: 10] |
| `-e, --gap-extend N` | Affine gap extend penalty for GA/SW [default: 1] |
| `-t, --threads N` | Number of threads (0 = auto) [default: 0] |
| `-z, --compression N` | HDF5 compression level (0-9) [default: 1] |
| `-f, --filter THRESHOLD` | Filter sequences with similarity above threshold |
| `-B, --benchmark` | Enable benchmarking mode |
| `-W, --no-write` | Disable writing to output file |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Display help message |

</details>

### Examples

```bash
# Run with default settings
./bin/main

# Use Smith-Waterman algorithm with 8 threads
./bin/main --align sw --threads 8

# Enable benchmarking mode with verbose output
./bin/main -B -v

# Custom paths
./bin/main -i dataset/avppred.csv -o results/avppred_results.h5
```

## Performance Benchmarks

<table>
  <tr>
    <th colspan="3">Test Environment</th>
  </tr>
  <tr>
    <td>Dataset</td>
    <td colspan="2">1042 sequences (lengths 6-49)</td>
  </tr>
  <tr>
    <td>Alignments</td>
    <td colspan="2">542,361 pairwise comparisons</td>
  </tr>
  <tr>
    <td>Algorithm</td>
    <td colspan="2">Needleman-Wunsch (score only)</td>
  </tr>
  <tr>
    <th>System</th>
    <th>Threads</th>
    <th>Alignment Time</th>
  </tr>
  <tr>
    <td rowspan="2">Linux (Arch)</td>
    <td>16</td>
    <td><strong>0.032s</strong></td>
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
    <td>Dataset</td>
    <td colspan="3">33,344 sequences (16x enlarged dataset)</td>
  </tr>
  <tr>
    <td>Alignments</td>
    <td colspan="3">555,894,496 pairwise comparisons</td>
  </tr>
  <tr>
    <td>Algorithm</td>
    <td colspan="3">Needleman-Wunsch (score only)</td>
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
    <td><strong>28.31s</strong></td>
    <td>5.85s</td>
  </tr>
</table>

> [!NOTE]
> - Processing speed: ~17-20 million alignments per second for my Ryzen 7 5700G
> - Smith-Waterman and Gotoh algorithms are slower than Needleman-Wunsch

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
</details>

## File Formats

- **Input**: CSV file with sequences
  - Automatically detects the column containing sequences
  - Asks for your input if it can't find the sequence column
  - Requires only one column with valid sequence data
  - For large datasets (10k-50k+ sequences), ensure you have sufficient RAM
  - Future updates may include chunked processing for memory-constrained environments
  
- **Output**: HDF5 file storing the similarity matrix
  - Matrix dimensions match the number of input sequences or filtered sequences if enabled
  - Compression level adjustable through command line options
  - Memory usage increases with the square of sequence count
  - Similar RAM restrictions as input

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