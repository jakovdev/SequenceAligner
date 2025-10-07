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
- Predefined substitution matrices
- HDF5 output format with optional compression

</details>

## Installation

### System Requirements

#### CPU Version
- Any x86-64 processor
- Enough RAM to store sequences from a dataset
- Disk space for output files (varies with dataset size)

#### CUDA Version
- NVIDIA GPU and drivers

### Releases

- ~~Download the latest release from [Releases](https://github.com/jakovdev/SequenceAligner/releases/latest)~~

> [!NOTE]
> Will release once it's more user friendly (GUI instead of command line)


### Building from source

<details>
<summary><strong>Linux</strong></summary>

#### Dependencies
- GCC, GNU Make
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
```
</details>

<details>
<summary><strong>Windows</strong></summary>

0. Download the project files using the green "Code" button on GitHub and select "Download ZIP". Extract the ZIP file to a folder of your choice. Take note of its location.

#### (RECOMMENDED) Windows MSVC (CUDA support)

1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (optional)
  - Required components: Runtime, Development

2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
  - Standalone packages: latest x64/x86 MSVC, vcpkg, any Windows 10/11 SDK
  - In the start menu, search for `x64 Native Tools Command Prompt for VS 2022` and open it

3. Navigate to the folder you downloaded the project using:
```bat
# Example:
cd C:\Users\John\Downloads\SequenceAligner
```

4. Install required libraries using vcpkg:
```bat
vcpkg install
```

5. Build the project using:
```bat
nmake /F NMakefile
```
6. You now have an executable in the `bin` folder, run with:
```bat
bin\seqalign.exe
```
> - Linux (both CUDA and CPU) provides the best and most consistent performance. The Windows CUDA build (MSVC) is supported and should run just as well. Windows CPU-only mode (MSVC) is not recommended - try the MSYS2 version below instead if you want to stay on Windows, but no guarantees.

#### Windows MSYS2 GCC (no CUDA support)

1. Install MSYS2 from https://www.msys2.org/
2. Open the MSYS2 UCRT64 terminal
3. Navigate to the folder you downloaded the project using:
```bash
# Example:
cd /c/Users/John/Downloads/SequenceAligner
```

> - Replace the folder path to the location you downloaded the project files
> - MSYS2 uses `/c/...` instead of `C:\...`

4. Install required tools by running:
```bash
./scripts/msys2_setup.sh
```

5. Build the project using:
```bash
mingw32-make
```

> - While this has faster CPU-only version than MSVC, for any serious usage, you should use the Linux version instead

</details>

## Usage

```bash
# Linux (CPU only version)
bin/seqalign [ARGUMENTS]

# Linux (CUDA+CPU version)
bin/seqalign-cuda [ARGUMENTS]

# Windows
bin\seqalign.exe [ARGUMENTS]
```

<details open>
<summary><strong>Command line arguments</strong></summary>

**Required arguments:**
| Argument | Description |
|--------|-------------|
| `-i, --input FILE` | Input FASTA or CSV file path |
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
| `-l, --list-matrices` | List all available substitution matrices |
| `-h, --help` | Display help message |

</details>

### Examples

> [!NOTE]
> - The input file is my own testing dataset provided with the codebase
> - This means all arguments are ones which work for that dataset (like sequence type)
> - You should change the arguments to match your dataset
> - Also, for relative file paths they should be relative to your current directory, not the binary location
> - Add `.exe` suffix on Windows to `seqalign`

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

# List all available substitution matrices
./bin/seqalign --list-matrices

# List all arguments
./bin/seqalign --help
```

## Performance Benchmarks

**Test Hardware:**
- **CPU**: AMD Ryzen 7 5700X3D (Zen 3 architecture, x86-64-v3)
- **RAM**: 2x16GB DDR4-3200 CL16
- **GPU**: NVIDIA GeForce RTX 4060 8GB with 1024 CUDA threads

### Dataset Characteristics

| **Dataset** | **Sequences** | **Avg. Length** | **Pairwise Alignments** | **Scale Category** |
|-------------|---------------|-----------------|------------------------|-------------------|
| AVPPred | 1,042 | 21.6 | 542,361 | Small |
| AMP | 9,409 | 30.5 | 44,259,936 | Medium |
| Drosophila | 58,746 | 17.9 | 1,725,516,885 | Large |

### CPU Threading Performance (AMP dataset)

| **Algorithm** | **1 Thread** | **4 Threads** | **8 Threads** | **16 Threads** |
|---------------|--------------|---------------|---------------|----------------|
| Needleman-Wunsch | 27.748s | 7.081s (3.92×) | 3.578s (7.76×) | 3.508s (7.91×) |
| Gotoh Affine | 166.204s | 42.118s (3.95×) | 21.244s (7.82×) | 13.305s (12.45×) |
| Smith-Waterman | 174.372s | 44.692s (3.90×) | 22.253s (7.84×) | 13.757s (12.68×) |

### CUDA Acceleration Performance

| **Dataset** | **Algorithm** | **CPU 16T Time (s) \| APS (M)** | **CUDA Time (s) \| APS (M)** | **Speedup** |
|-------------|---------------|------------------------------|-------------------------------|-------------|
| **AVPPred** | NW | 0.024 \| 22.949 | 0.009 \| 58.223 | **2.54×** |
| | GA | 0.084 \| 6.440 | 0.011 \| 49.418 | **7.67×** |
| | SW | 0.088 \| 6.130 | 0.011 \| 50.316 | **8.21×** |
| **AMP** | NW | 3.508 \| 12.616 | 0.670 \| 66.049 | **5.23×** |
| | GA | 13.305 \| 3.327 | 1.610 \| 27.486 | **8.26×** |
| | SW | 13.757 \| 3.217 | 1.602 \| 27.623 | **8.59×** |
| **Drosophila** | NW | 51.340 \| 33.610 | 21.350 \| 80.820 | **2.40×** |
| | GA | 180.672 \| 9.551 | 34.201 \| 50.452 | **5.28×** |
| | SW | 185.188 \| 9.318 | 36.197 \| 47.670 | **5.12×** |

*APS = Alignments per second (millions)*

> [!NOTE]
> For a large dataset like Drosophila, there was an issue during benchmark which lead to ~50% GPU utilization. After fixing it, the actual APS should be around 120-125M for NW, with times around 13-14 seconds.
> The issue was related to the double buffer implementation which was erroneously blocking processing when copying results from the GPU, which halved the processing speed. This has been fixed in the latest code with asynchronous copying.
> The table will be updated soon with new times soon(tm).

### Performance Summary

- **Sequence Length Impact**: Shorter sequences significantly improve throughput due to reduced computational complexity per alignment
- **Algorithm Scaling**: Complex algorithms (Gotoh, Smith-Waterman) benefit more from both CPU threading and GPU acceleration
- **CUDA Advantages**: GPU acceleration is most effective for affine gap penalty algorithms and datasets with moderate to long sequences in an all-vs-all context where "long" is 30+ average amino acids or nucleotides
- **Optimal Performance**: Drosophila dataset achieves highest absolute throughput (~~80.82M~~ 124.9M alignments/sec) with CUDA Needleman-Wunsch due to optimal combination of short sequences and massive scale

> [!NOTE]
> - For very large datasets where the Similarity Matrix exceeds available RAM/VRAM, alignments are performed in batches and written to disk before HDF5 conversion
> - CPU implementations remain valuable for systems without GPU acceleration or specific memory constraints
> - For datasets with sequences longer that 1024 amino acids or nucleotides, try editing cuda_kernels.cu to increase the `MAX_CUDA_SEQUENCE_LENGTH` constant to match your dataset and recompile. However, if memory issues arise, you might need to fall back to the CPU version instead.

## Implementation Details

<details>
<summary><strong>Algorithm Implementations</strong></summary>

- **Needleman-Wunsch**: Global alignment with linear gap penalties
- **Smith-Waterman**: Local alignment with affine gap penalties 
- **Gotoh Algorithm**: Global alignment with affine gap penalties

All implementations use dynamic programming with optimized matrix operations.

> Parasail python equivalents
> - parasail.nw() is actually the Gotoh algorithm with affine gaps in SequenceAligner
> - To get actual linear gaps in Parasail you need to set the `open` and `extend` parameters to the same value
> - This also applies to the Gotoh algorithm in SequenceAligner, but you should use NW since it is faster and takes one value
> - **Needleman-Wunsch**: **Parasail:** `parasail.nw(..., open=gap, extend=gap, ...)`,  **SequenceAligner:** `-a nw -p gap`
> - **Smith-Waterman**: **Parasail:** `parasail.sw(..., open=gap_open, extend=gap_extend, ...)`, **SequenceAligner:** `-a sw -s gap_open -e gap_extend`
> - **Gotoh Algorithm**: **Parasail:** `parasail.nw(..., open=gap_open, extend=gap_extend, ...)`, **SequenceAligner:** `-a ga -s gap_open -e gap_extend`

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
- **File Type**: FASTA or CSV format
- **Sequences**:
  - For protein sequences: IUPAC amino acid single letter codes (ARNDCQEGHILKMFPSTWYVBZX*)
  - For nucleotide sequences: IUPAC nucleotide single letter codes (ATGCSWRYKMBVHDN*)
- **CSV Requirements**:
  - Must contain one column with biological sequence data (amino acids or nucleotides)
  - No specific header requirements - the program will scan and identify the sequence column
  - The program will prompt you to select the correct one if it can't find one automatically

### Output Format
- **File Type**: HDF5 (.h5) - a common scientific data format
- **Content**:
  - Contains a similarity matrix where each cell represents the alignment score between sequence pairs
  - Also stores the original or filtered sequences and their lengths
- **Size Considerations**:
  - The matrix grows with the square of the sequence count (1,000 sequences = 1 million cells = 4 MB)
    > Each score number (cell) is 4 bytes
  - For very large datasets, the program will automatically use disk-based storage when needed, so check if you have enough free disk storage if aligning a dataset with hundreds of thousands of sequences (100+ GB simlarity matrix!)
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