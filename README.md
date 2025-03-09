# Sequence Aligner

A sequence alignment tool for academic purposes. Not accepting contributions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Formats](#default-file-formats)
- [Performance](#performance-benchmarks)
- [Development](#development)

## Overview
Sequence Aligner provides efficient DNA/protein sequence alignment using memory-mapped I/O and AVX optimizations. Supports both single and multi-threaded operations.

## Features
- Needleman-Wunsch algorithm
- Similarity analysis
- Memory mapped file I/O
- Multithreading support
- Cross-platform (Windows/Linux)
- Simple interface
- Dark mode

## Requirements

<details open>
<summary>Windows</summary>

- Windows 10 or later
- 64-bit system
</details>

<details>
<summary>Linux</summary>

```sh 
# Core dependencies (use distro equivalent package manager and package if not specified)
sudo apt install gcc make python3-tk
sudo pacman -S gcc make python tk

# Optional: Dark mode
sudo apt install python3-pip
sudo pacman -S python-pip
pip install customtkinter

# Optional: Windows build support (useful for WSL2)
sudo apt install gcc-mingw-w64
sudo pacman -S mingw-w64-gcc

# Install if you encounter font problems when starting the program
sudo apt install ttf-dejavu
sudo pacman -S ttf-dejavu
```
### WSL2 only: Install an X server for GUI support
1. Install [VcXsrv](https://github.com/marchaesen/vcxsrv/releases/latest)
2. Start XLaunch
3. Use the following settings:
	- Multiple windows (default)
	- Display number: 0
	- Start no client (default)
	- Disable access control
	- Save configuration (to avoid setting up every time)
4. Execute the following one-line command in WSL2:
```sh
echo 'export DISPLAY=$(ip route show default | awk "{print \$3}"):0.0' >> ~/.bashrc | source ~/.bashrc
```
> Replace `~/.bashrc` with `~/.zshrc` if using zsh
>
> This will set the display environment variable every time you start WSL2
>
> Explanation: `ip route show default` gets the default gateway, `awk "{print \$3}"` extracts the IP address of the WSL2 Hyper-V adapter, and `DISPLAY=$(...)` sets the display environment variable
5. The X server must be running before starting the program
</details>

## Installation

<details open>
<summary>Windows</summary>

1. Download the latest installer from [Releases](https://github.com/jakovdev/SequenceAligner/releases/latest)
2. Run `SequenceAligner-Windows-Setup.exe`
3. Follow the installation wizard

> ℹ️ **Note**
> If you get Windows Defender popups and unsure about security, you can install [python](https://www.python.org/downloads/) and click the green `< > Code` `->` `Download ZIP`, unzip and then open a Terminal inside the SequenceAligner folder, then:
> ```sh
> # Optional: Dark mode
> pip install customtkinter
> 
> # Starts the application
> python start.py
> ```
>
> The only reason Defender triggers is because I have to pay a large fee to Microsoft so that they can sign the software.
>
> You might also get additional popups when running the program. They pop up because the program is building executables using your settings (format, csv sizes etc.), and then runs them (which too are not signed by Microsoft), hence the popups.
>
> Last resort is excluding the folder from Defender, but that is not recommended.
>
> I might look into different solutions in the future to try to mitigate this issue.
</details>

<details>
<summary>Linux</summary>

```sh
# Clone repository
git clone https://github.com/jakovdev/SequenceAligner.git
cd SequenceAligner

# Run the launcher
python3 start.py
```
</details>

## Usage

<details open>
<summary>Windows</summary>

1. Launch SequenceAligner from the Start Menu or desktop shortcut
2. Select options:
	- **Run**: Start alignment and create the output file
	- **Settings**: Modify format, I/O files, and parameters
3. Click `Save` to apply changes or `Reset` for defaults
4. Use `Tuning` for batch size optimization
</details>

<details>
<summary>Linux</summary>

1. Launch start.py - `python3 start.py`
2. Follow the same steps as Windows
</details>

## Default File Formats

<details>
<summary>Input CSV Format</summary>

```csv
sequence,label
KPVSLS,0
LNNSRA,0
```
- Standard dataset: [avpdb.csv](testing/datasets/avpdb.csv) (1042 lines, 27KB)
- Large dataset: Generated with this [script](scripts/create_mega_dataset.py) (4,001,280 lines, 97.6MB)
</details>

<details>
<summary>Output CSV Format</summary>

```csv
sequence1,sequence2,label1,label2,score,alignment,matches,mismatches,gaps,similarity
KPVSLS,LNNSRA,0,0,-5,"('KPVSLS', 'LNNSRA')",1,5,0,16.66%
```
- Output: [results.csv](results/results.csv)
</details>

## Performance Benchmarks

| Mode | Platform | Without Write | With Write |
|------|----------|--------------|------------|
| Singlethreaded* | Windows | ~500µs | ~750µs |
| Singlethreaded* | WSL2 | ~360µs | ~550µs |
| Multithreaded** | Windows | ~450ms | ~800ms |
| Multithreaded** | WSL2 | ~770ms | 1-2s |

> ℹ️ **Notes**
>
> \* Using standard dataset
>
> \** Using large dataset
>
> *Timing excludes initialization overhead*

## [Development](TODO.md)

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

This software is intended for academic purposes only. If you use this software in your research, please cite:

```bibtex
@software{SequenceAligner,
  author = {jakovdev},
  title = {Sequence Aligner},
  year = {2025},
  url = {https://github.com/jakovdev/SequenceAligner}
}
```
