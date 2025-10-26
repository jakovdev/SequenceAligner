#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import time
import statistics
from pathlib import Path
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    import plotly.offline as pyo
    from plotly.colors import qualitative

    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. Only text tables will be generated.")
    print("Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

BINARY = "bin/seqalign"
# BINARY = "script/seqalign_parasail.py" Remember to uncomment "CUDA" in EXECUTION_MODES

NUM_RUNS = 1  # if > 1, statistical analysis is performed
NUM_CUDA_THREADS = 1024

INPUT_FILES = ["datasets/avppred.csv", "datasets/amp.csv"]

ALIGNMENT_METHODS = ["nw", "ga", "sw"]

EXECUTION_MODES = {
    "CPU-01T": ["-T", "1", "-C"],
    "CPU-04T": ["-T", "4", "-C"],
    "CPU-08T": ["-T", "8", "-C"],
    "CPU-16T": ["-T", "16", "-C"],
    "CUDA": [],  # Comment if using Parasail binary, if leftover it will execute CPU-01T
}

BASELINE_MODE = "CPU-01T"

CONSTANT_ARGS = ["-t", "amino", "-m", "blosum62", "-D", "-B"]

GAP_PENALTIES = {
    "nw": ["-p", "4"],
    "ga": ["-s", "10", "-e", "1"],
    "sw": ["-s", "10", "-e", "1"],
}

RESULTS_DIR = "benchmark_results"
H5_RESULTS_DIR = "results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class BenchmarkRun:
    dataset: str
    method: str
    mode: str
    run_number: int

    compute_time: float = 0.0
    io_time: float = 0.0
    total_time: float = 0.0
    wall_time: float = 0.0

    alignments_per_second: float = 0.0
    alignments_per_second_per_thread: float = 0.0
    avg_time_per_thread: float = 0.0

    num_sequences: int = 0
    num_alignments: int = 0
    avg_sequence_length: float = 0.0

    num_threads: int = 1
    cuda_enabled: bool = False
    matrix_checksum: int = 0

    success: bool = False
    error_message: str = ""


@dataclass
class BenchmarkStats:
    dataset: str
    method: str
    mode: str

    compute_time_mean: float = 0.0
    compute_time_std: float = 0.0
    compute_time_min: float = 0.0
    compute_time_max: float = 0.0

    io_time_mean: float = 0.0
    io_time_std: float = 0.0

    total_time_mean: float = 0.0
    total_time_std: float = 0.0

    aps_mean: float = 0.0
    aps_std: float = 0.0
    aps_min: float = 0.0
    aps_max: float = 0.0

    num_sequences: int = 0
    num_alignments: int = 0
    avg_sequence_length: float = 0.0
    num_threads: int = 1
    cuda_enabled: bool = False

    num_successful_runs: int = 0
    coefficient_of_variation: float = 0.0


class BenchmarkManager:
    def __init__(self):
        self.results_dir = Path(RESULTS_DIR)
        self.results_dir.mkdir(exist_ok=True)

        self.all_runs: List[BenchmarkRun] = []
        self.all_stats: List[BenchmarkStats] = []

        self.current_dataset_runs: List[BenchmarkRun] = []
        self.current_dataset_stats: List[BenchmarkStats] = []
        self.current_dataset_name: str = ""

    def start_dataset(self, dataset_name: str):
        self.current_dataset_name = dataset_name
        self.current_dataset_runs = []
        self.current_dataset_stats = []
        print(f"\nStarting dataset: {dataset_name.upper()}")
        print("=" * 60)

    def add_benchmark_result(self, runs: List[BenchmarkRun], stats: BenchmarkStats):
        self.current_dataset_runs.extend(runs)
        self.current_dataset_stats.append(stats)

        self.all_runs.extend(runs)
        self.all_stats.append(stats)

        self._update_dataset_reports()
        self._update_combined_reports()

    def _update_dataset_reports(self):
        if not self.current_dataset_stats:
            return

        dataset_name = self.current_dataset_name

        text_report = self._generate_text_report(
            self.current_dataset_stats, f"Dataset: {dataset_name.upper()}"
        )

        text_file = self.results_dir / f"benchmark_{dataset_name}_{TIMESTAMP}.txt"
        with open(text_file, "w") as f:
            f.write(text_report)

        if PLOTLY_AVAILABLE:
            html_report = self._generate_html_report(
                self.current_dataset_stats,
                self.current_dataset_runs,
                f"Benchmark Report: {dataset_name.upper()}",
            )
            html_file = self.results_dir / f"benchmark_{dataset_name}_{TIMESTAMP}.html"
            with open(html_file, "w") as f:
                f.write(html_report)

        raw_data = {
            "metadata": {
                "dataset": dataset_name,
                "timestamp": TIMESTAMP,
                "num_runs_per_config": NUM_RUNS,
                "baseline_mode": BASELINE_MODE if not is_single_run() else None,
                "completed_configurations": len(self.current_dataset_stats),
                "successful_configurations": len(
                    [s for s in self.current_dataset_stats if s.num_successful_runs > 0]
                ),
            },
            "statistics": [asdict(stat) for stat in self.current_dataset_stats],
            "raw_runs": [asdict(run) for run in self.current_dataset_runs],
        }

        json_file = self.results_dir / f"benchmark_{dataset_name}_{TIMESTAMP}.json"
        with open(json_file, "w") as f:
            json.dump(raw_data, f, indent=2)

    def _update_combined_reports(self):
        if not self.all_stats:
            return

        text_report = self._generate_text_report(self.all_stats, "BENCHMARK REPORT")

        text_file = self.results_dir / f"benchmark_{TIMESTAMP}.txt"
        with open(text_file, "w") as f:
            f.write(text_report)

        if PLOTLY_AVAILABLE:
            html_report = self._generate_html_report(
                self.all_stats, self.all_runs, "Benchmark Report"
            )
            html_file = self.results_dir / f"benchmark_{TIMESTAMP}.html"
            with open(html_file, "w") as f:
                f.write(html_report)

        raw_data = {
            "metadata": {
                "timestamp": TIMESTAMP,
                "num_runs_per_config": NUM_RUNS,
                "baseline_mode": BASELINE_MODE if not is_single_run() else None,
                "total_configurations": len(self.all_stats),
                "successful_configurations": len(
                    [s for s in self.all_stats if s.num_successful_runs > 0]
                ),
                "datasets_processed": len(set(s.dataset for s in self.all_stats)),
            },
            "statistics": [asdict(stat) for stat in self.all_stats],
            "raw_runs": [asdict(run) for run in self.all_runs],
        }

        json_file = self.results_dir / f"benchmark_combined_{TIMESTAMP}.json"
        with open(json_file, "w") as f:
            json.dump(raw_data, f, indent=2)

    def _generate_text_report(
        self, stats_list: List[BenchmarkStats], title: str
    ) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(
            f"{'Run' if is_single_run() else 'Runs'} per configuration: {NUM_RUNS}"
        )
        if not is_single_run():
            lines.append(f"Baseline mode for speedup: {BASELINE_MODE}")
        lines.append("")

        successful_configs = len([s for s in stats_list if s.num_successful_runs > 0])
        total_configs = len(stats_list)

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total configurations: {total_configs}")
        lines.append(f"Successful configurations: {successful_configs}")
        lines.append(f"Success rate: {successful_configs/total_configs*100:.1f}%")
        lines.append("")

        datasets = sorted(set(s.dataset for s in stats_list))

        for dataset in datasets:
            dataset_stats = [
                s
                for s in stats_list
                if s.dataset == dataset and s.num_successful_runs > 0
            ]
            if not dataset_stats:
                continue

            lines.append(f"DATASET: {dataset.upper()}")
            lines.append("-" * 60)

            first_stat = dataset_stats[0]
            lines.append(f"Sequences: {first_stat.num_sequences:,}")
            lines.append(f"Pairwise alignments: {first_stat.num_alignments:,}")
            lines.append(
                f"Average sequence length: {first_stat.avg_sequence_length:.1f}"
            )
            lines.append("")

            lines.append("Performance Results:")
            lines.append("")

            if is_single_run():
                header = (
                    f"{'Algorithm':<12} {'Mode':<12} {'Compute(s)':<12} {'APS':<12}"
                )
            else:
                header = f"{'Algorithm':<12} {'Mode':<12} {'Compute(s)':<12} {'±':<8} {'APS':<12} {'±':<10} {'CV%':<6}"

            lines.append(header)
            lines.append("-" * len(header))

            for stat in sorted(dataset_stats, key=lambda x: (x.method, x.mode)):
                if is_single_run():
                    lines.append(
                        f"{stat.method.upper():<12} {stat.mode:<12} "
                        f"{stat.compute_time_mean:<12.3f} "
                        f"{stat.aps_mean:<12.0f}"
                    )
                else:
                    lines.append(
                        f"{stat.method.upper():<12} {stat.mode:<12} "
                        f"{stat.compute_time_mean:<12.3f} {stat.compute_time_std:<8.3f} "
                        f"{stat.aps_mean:<12.0f} {stat.aps_std:<10.0f} "
                        f"{stat.coefficient_of_variation:<6.1f}"
                    )

            lines.append("")
            lines.append("")

        lines.append("PERFORMANCE HIGHLIGHTS")
        lines.append("-" * 40)

        if stats_list:
            fastest = max(
                stats_list, key=lambda x: x.aps_mean if x.num_successful_runs > 0 else 0
            )
            if fastest.num_successful_runs > 0:
                lines.append(f"Highest throughput: {fastest.aps_mean:,.0f} APS")
                lines.append(
                    f"  {fastest.dataset} | {fastest.method.upper()} | {fastest.mode}"
                )
                lines.append("")

            if not is_single_run():
                reliable_stats = [
                    s
                    for s in stats_list
                    if s.num_successful_runs > 0 and s.coefficient_of_variation > 0
                ]
                if reliable_stats:
                    most_reliable = min(
                        reliable_stats, key=lambda x: x.coefficient_of_variation
                    )
                    lines.append(
                        f"Most consistent: {most_reliable.coefficient_of_variation:.1f}% CV"
                    )
                    lines.append(
                        f"  {most_reliable.dataset} | {most_reliable.method.upper()} | {most_reliable.mode}"
                    )
                    lines.append("")

        return "\n".join(lines)

    def _generate_html_report(
        self, stats_list: List[BenchmarkStats], all_runs: List[BenchmarkRun], title: str
    ) -> str:
        if not PLOTLY_AVAILABLE:
            return ""

        performance_fig = create_performance_comparison_plot(stats_list)
        summary_table_fig = create_detailed_summary_table(stats_list)

        performance_html = pyo.plot(
            performance_fig, output_type="div", include_plotlyjs=False
        )
        table_html = pyo.plot(
            summary_table_fig, output_type="div", include_plotlyjs=False
        )

        total_runs = len(all_runs)
        successful_runs = len([r for r in all_runs if r.success])
        datasets_tested = len(set(r.dataset for r in all_runs))
        methods_tested = len(set(r.method for r in all_runs))
        modes_tested = len(set(r.mode for r in all_runs))

        if is_single_run():
            stats_description = "Single run per configuration for quick analysis."
        else:
            stats_description = f"Statistical analysis based on {NUM_RUNS} runs per configuration for reliability."

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    color: #333;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                }}
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #667eea;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .stat-label {{
                    font-size: 0.9em;
                    color: #666;
                    margin-top: 5px;
                }}
                .section {{
                    margin: 40px 0;
                }}
                .section h2 {{
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                .footer {{
                    margin-top: 50px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    text-align: center;
                    font-size: 0.9em;
                    color: #666;
                }}
                .update-time {{
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    background: rgba(102, 126, 234, 0.9);
                    color: white;
                    padding: 5px 10px;
                    border-radius: 5px;
                    font-size: 0.8em;
                }}
            </style>
        </head>
        <body>
            <div class="update-time">
                Last updated: {datetime.now().strftime('%H:%M:%S')}
            </div>
            
            <div class="header">
                <h1>{title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Baseline mode for speedup analysis: {BASELINE_MODE}</p>
            </div>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-value">{successful_runs}/{total_runs}</div>
                    <div class="stat-label">Successful Runs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{datasets_tested}</div>
                    <div class="stat-label">Datasets Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{methods_tested}</div>
                    <div class="stat-label">Algorithms Tested</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{modes_tested}</div>
                    <div class="stat-label">Execution Modes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{NUM_RUNS}</div>
                    <div class="stat-label">{'Run' if is_single_run() else 'Runs'} per Configuration</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Analysis</h2>
                <p>Comprehensive comparison of algorithms across different execution modes. Throughput uses logarithmic scale due to quadratic scaling of alignments (n² from n sequences). Speedup analysis uses {BASELINE_MODE} as baseline for practical comparison.</p>
                {performance_html}
            </div>
            
            <div class="section">
                <h2>Detailed Results</h2>
                <p>Complete benchmark results{'.' if is_single_run() else ' with statistical measures. CV% indicates coefficient of variation for compute time reliability.'}</p>
                {table_html}
            </div>
            
            <div class="footer">
                <p>Generated by Sequence Aligner Benchmark</p>
                <p>{stats_description}</p>
            </div>
        </body>
        </html>
        """

        return html_content


def in_project_root():
    current_dir = Path.cwd()

    if current_dir.name == "script":
        os.chdir(current_dir.parent)

    if not (Path("bin").exists() and Path("datasets").exists()):
        print(
            "Error: Could not find project structure (bin/ and datasets/ directories)"
        )
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)


def get_binary_path() -> str:
    if not Path(BINARY).exists():
        if Path(f"{BINARY}.exe").exists():
            return f"{BINARY}.exe"
        else:
            raise FileNotFoundError(f"Binary not found: {BINARY}")

    return BINARY


def is_python_script(binary_path: str) -> bool:
    return binary_path.endswith(".py")


def extract_dataset_name(input_file: str) -> str:
    return Path(input_file).stem


def generate_output_path(input_file: str) -> str:
    dataset_name = extract_dataset_name(input_file)
    results_dir = Path(H5_RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    return str(results_dir / f"{dataset_name}.h5")


def extract_thread_count(mode: str) -> int:
    if mode == "CUDA":
        return NUM_CUDA_THREADS
    match = re.search(r"(\d+)T", mode)
    return int(match.group(1)) if match else 1


def is_single_run() -> bool:
    return NUM_RUNS == 1


def parse_program_output(output: str) -> Dict[str, Any]:
    data = {}

    # Performance timing
    compute_match = re.search(r"Compute:\s*([\d.]+)\s*sec", output)
    if compute_match:
        data["compute_time"] = float(compute_match.group(1))

    io_match = re.search(r"I/O:\s*([\d.]+)\s*sec", output)
    if io_match:
        data["io_time"] = float(io_match.group(1))

    total_match = re.search(r"Total:\s*([\d.]+)\s*sec", output)
    if total_match:
        data["total_time"] = float(total_match.group(1))

    # Throughput metrics
    aps_match = re.search(r"Alignments per second:\s*([\d.]+)", output)
    if aps_match:
        data["alignments_per_second"] = float(aps_match.group(1))

    aps_per_thread_match = re.search(
        r"Alignments per second per thread:\s*([\d.]+)", output
    )
    if aps_per_thread_match:
        data["alignments_per_second_per_thread"] = float(aps_per_thread_match.group(1))

    avg_thread_time_match = re.search(
        r"Average time per thread:\s*([\d.]+)\s*sec", output
    )
    if avg_thread_time_match:
        data["avg_time_per_thread"] = float(avg_thread_time_match.group(1))

    # Dataset characteristics
    seq_match = re.search(r"Found (\d+) sequences", output)
    if seq_match:
        data["num_sequences"] = int(seq_match.group(1))

    align_match = re.search(r"Will perform (\d+) pairwise alignments", output)
    if align_match:
        data["num_alignments"] = int(align_match.group(1))

    avg_len_match = re.search(r"Average sequence length:\s*([\d.]+)", output)
    if avg_len_match:
        data["avg_sequence_length"] = float(avg_len_match.group(1))

    # System configuration
    threads_match = re.search(r"Threads:\s*(\d+)", output)
    if threads_match:
        data["num_threads"] = int(threads_match.group(1))

    cuda_match = re.search(r"CUDA:\s*(Enabled|Disabled)", output)
    if cuda_match:
        data["cuda_enabled"] = cuda_match.group(1) == "Enabled"

    if data.get("cuda_enabled", False):
        data["num_threads"] = NUM_CUDA_THREADS

    # Matrix checksum for verification
    checksum_match = re.search(r"Matrix checksum:\s*(-?\d+)", output)
    if checksum_match:
        data["matrix_checksum"] = int(checksum_match.group(1))

    return data


def run_single_benchmark(
    input_file: str, method: str, mode: str, mode_args: List[str], run_number: int
) -> BenchmarkRun:
    dataset_name = extract_dataset_name(input_file)
    output_file = generate_output_path(input_file)

    binary = get_binary_path()

    if is_python_script(binary):
        cmd = (
            ["python", binary, "-i", input_file, "-o", output_file, "-a", method]
            + CONSTANT_ARGS
            + GAP_PENALTIES[method]
            + mode_args
        )
    else:
        cmd = (
            [binary, "-i", input_file, "-o", output_file, "-a", method]
            + CONSTANT_ARGS
            + GAP_PENALTIES[method]
            + mode_args
        )

    run = BenchmarkRun(
        dataset=dataset_name, method=method, mode=mode, run_number=run_number
    )

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        end_time = time.time()

        run.wall_time = end_time - start_time

        if result.returncode != 0:
            run.error_message = result.stderr
            return run

        parsed_data = parse_program_output(result.stdout)

        for key, value in parsed_data.items():
            if hasattr(run, key):
                setattr(run, key, value)

        run.success = True

    except subprocess.TimeoutExpired:
        run.error_message = "Timeout (3600s exceeded)"
    except Exception as e:
        run.error_message = str(e)

    return run


def run_benchmarks(
    input_file: str, method: str, mode: str, mode_args: List[str]
) -> List[BenchmarkRun]:
    runs = []
    dataset_name = extract_dataset_name(input_file)

    if is_single_run():
        print(f"  Running benchmark for {dataset_name}|{method}|{mode}")
    else:
        print(f"  Running {NUM_RUNS} iterations for {dataset_name}|{method}|{mode}")

    for i in range(NUM_RUNS):
        if is_single_run():
            print(f"    Executing...", end=" ")
        else:
            print(f"    Run {i+1}/{NUM_RUNS}...", end=" ")

        run = run_single_benchmark(input_file, method, mode, mode_args, i + 1)
        runs.append(run)

        if run.success:
            print(f"✓ {run.compute_time:.3f}s ({run.alignments_per_second:.0f} APS)")
        else:
            print(f"✗ {run.error_message}")

    return runs


def calculate_statistics(runs: List[BenchmarkRun]) -> BenchmarkStats:
    successful_runs = [r for r in runs if r.success]

    if not successful_runs:
        return BenchmarkStats(
            dataset=runs[0].dataset,
            method=runs[0].method,
            mode=runs[0].mode,
            num_successful_runs=0,
        )

    compute_times = [r.compute_time for r in successful_runs]
    io_times = [r.io_time for r in successful_runs]
    total_times = [r.total_time for r in successful_runs]
    aps_values = [r.alignments_per_second for r in successful_runs]

    first_run = successful_runs[0]

    stats = BenchmarkStats(
        dataset=first_run.dataset,
        method=first_run.method,
        mode=first_run.mode,
        compute_time_mean=statistics.mean(compute_times),
        compute_time_std=(
            statistics.stdev(compute_times) if len(compute_times) > 1 else 0.0
        ),
        compute_time_min=min(compute_times),
        compute_time_max=max(compute_times),
        io_time_mean=statistics.mean(io_times),
        io_time_std=statistics.stdev(io_times) if len(io_times) > 1 else 0.0,
        total_time_mean=statistics.mean(total_times),
        total_time_std=statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
        aps_mean=statistics.mean(aps_values),
        aps_std=statistics.stdev(aps_values) if len(aps_values) > 1 else 0.0,
        aps_min=min(aps_values),
        aps_max=max(aps_values),
        num_sequences=first_run.num_sequences,
        num_alignments=first_run.num_alignments,
        avg_sequence_length=first_run.avg_sequence_length,
        num_threads=first_run.num_threads,
        cuda_enabled=first_run.cuda_enabled,
        num_successful_runs=len(successful_runs),
        coefficient_of_variation=(
            (statistics.stdev(compute_times) / statistics.mean(compute_times) * 100)
            if len(compute_times) > 1 and statistics.mean(compute_times) > 0
            else 0.0
        ),
    )

    return stats


def create_performance_comparison_plot(stats_list: List[BenchmarkStats]) -> go.Figure:
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Compute Time (log₁₀ scale) by Algorithm & Mode",
            "Throughput (log₂ scale) by Algorithm & Mode",
            f"Parallel Speedup (baseline: {BASELINE_MODE})",
            "Algorithm Performance by Sequence Characteristics",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    colors = qualitative.Set1
    method_colors = {
        method: colors[i % len(colors)] for i, method in enumerate(ALIGNMENT_METHODS)
    }

    datasets = sorted(set(s.dataset for s in stats_list))
    modes = sorted(set(s.mode for s in stats_list))

    # Plot 1: Compute Time Comparison
    for method in ALIGNMENT_METHODS:
        for dataset in datasets:
            x_vals = []
            y_vals = []
            error_vals = []

            for mode in modes:
                stat = next(
                    (
                        s
                        for s in stats_list
                        if s.dataset == dataset
                        and s.method == method
                        and s.mode == mode
                    ),
                    None,
                )
                if stat and stat.num_successful_runs > 0:
                    x_vals.append(f"{mode}")
                    y_vals.append(stat.compute_time_mean)
                    if not is_single_run():
                        error_vals.append(stat.compute_time_std)

            if x_vals:
                trace_kwargs = {
                    "x": x_vals,
                    "y": y_vals,
                    "name": f"{method.upper()}-{dataset}",
                    "marker_color": method_colors[method],
                    "opacity": 0.7,
                    "showlegend": True,
                }

                if not is_single_run() and error_vals:
                    trace_kwargs["error_y"] = dict(type="data", array=error_vals)

                fig.add_trace(go.Bar(**trace_kwargs), row=1, col=1)

    # Plot 2: Throughput Comparison
    for method in ALIGNMENT_METHODS:
        for dataset in datasets:
            x_vals = []
            y_vals = []
            error_vals = []

            for mode in modes:
                stat = next(
                    (
                        s
                        for s in stats_list
                        if s.dataset == dataset
                        and s.method == method
                        and s.mode == mode
                    ),
                    None,
                )
                if stat and stat.num_successful_runs > 0:
                    x_vals.append(f"{mode}")
                    y_vals.append(stat.aps_mean)
                    if not is_single_run():
                        error_vals.append(stat.aps_std)

            if x_vals:
                trace_kwargs = {
                    "x": x_vals,
                    "y": y_vals,
                    "name": f"{method.upper()}-{dataset}",
                    "marker_color": method_colors[method],
                    "opacity": 0.7,
                    "showlegend": False,
                }

                if not is_single_run() and error_vals:
                    trace_kwargs["error_y"] = dict(type="data", array=error_vals)

                fig.add_trace(go.Bar(**trace_kwargs), row=1, col=2)

    # Plot 3: Parallel Speedup Analysis
    cpu_modes = [mode for mode in modes if mode.startswith("CPU")]
    for method in ALIGNMENT_METHODS:
        for dataset in datasets:
            thread_counts = []
            speedups = []

            baseline_stat = next(
                (
                    s
                    for s in stats_list
                    if s.dataset == dataset
                    and s.method == method
                    and s.mode == BASELINE_MODE
                ),
                None,
            )

            if baseline_stat and baseline_stat.num_successful_runs > 0:
                baseline_time = baseline_stat.compute_time_mean

                for mode in cpu_modes:
                    stat = next(
                        (
                            s
                            for s in stats_list
                            if s.dataset == dataset
                            and s.method == method
                            and s.mode == mode
                        ),
                        None,
                    )
                    if stat and stat.num_successful_runs > 0:
                        threads = extract_thread_count(mode)
                        speedup = baseline_time / stat.compute_time_mean
                        thread_counts.append(threads)
                        speedups.append(speedup)

                if thread_counts:
                    fig.add_trace(
                        go.Scatter(
                            x=thread_counts,
                            y=speedups,
                            mode="lines+markers",
                            name=f"{method.upper()}-{dataset}",
                            line=dict(color=method_colors[method]),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

    # Ideal speedup line (from baseline)
    if cpu_modes:
        baseline_threads = extract_thread_count(BASELINE_MODE)
        max_threads = max([extract_thread_count(mode) for mode in cpu_modes])
        ideal_x = list(range(baseline_threads, max_threads + 1))
        ideal_y = [t / baseline_threads for t in ideal_x]

        fig.add_trace(
            go.Scatter(
                x=ideal_x,
                y=ideal_y,
                mode="lines",
                name="Ideal Speedup",
                line=dict(dash="dash", color="black"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Plot 4: Algorithm Performance vs Sequence Characteristics
    for method in ALIGNMENT_METHODS:
        seq_lengths = []
        throughputs = []
        dataset_names = []

        for dataset in datasets:
            # Try to find CPU-16T first, then any available mode
            stat = next(
                (
                    s
                    for s in stats_list
                    if s.dataset == dataset
                    and s.method == method
                    and s.mode == "CPU-16T"
                ),
                None,
            )
            if not stat:
                stat = next(
                    (
                        s
                        for s in stats_list
                        if s.dataset == dataset and s.method == method
                    ),
                    None,
                )

            if stat and stat.num_successful_runs > 0:
                seq_lengths.append(stat.avg_sequence_length)
                throughputs.append(stat.aps_mean)
                dataset_names.append(dataset)

        if seq_lengths:
            fig.add_trace(
                go.Scatter(
                    x=seq_lengths,
                    y=throughputs,
                    mode="markers+text",
                    text=dataset_names,
                    textposition="top center",
                    name=f"{method.upper()}",
                    marker=dict(color=method_colors[method], size=12),
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    fig.update_layout(
        title_text="Performance Analysis",
        title_x=0.5,
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="Execution Mode", row=1, col=1)
    fig.update_yaxes(title_text="Compute Time (s)", type="log", row=1, col=1)

    fig.update_xaxes(title_text="Execution Mode", row=1, col=2)
    fig.update_yaxes(title_text="Alignments/second", type="log", row=1, col=2)

    fig.update_xaxes(title_text="Number of Threads", row=2, col=1)
    fig.update_yaxes(title_text="Speedup Factor", row=2, col=1)

    fig.update_xaxes(title_text="Average Sequence Length", row=2, col=2)
    fig.update_yaxes(title_text="Throughput (APS)", type="log", row=2, col=2)

    return fig


def create_detailed_summary_table(stats_list: List[BenchmarkStats]) -> go.Figure:
    if is_single_run():
        headers = [
            "Dataset",
            "Algorithm",
            "Mode",
            "Sequences",
            "Alignments",
            "Avg Seq Len",
            "Threads",
            "CUDA",
            "Compute Time (s)",
            "Throughput (APS)",
        ]
    else:
        headers = [
            "Dataset",
            "Algorithm",
            "Mode",
            "Sequences",
            "Alignments",
            "Avg Seq Len",
            "Threads",
            "CUDA",
            "Compute Time (s)",
            "±",
            "Throughput (APS)",
            "±",
            "CV%",
            "Runs",
        ]

    rows = []
    for stat in stats_list:
        if stat.num_successful_runs > 0:
            if is_single_run():
                rows.append(
                    [
                        stat.dataset,
                        stat.method.upper(),
                        stat.mode,
                        f"{stat.num_sequences:,}",
                        f"{stat.num_alignments:,}",
                        f"{stat.avg_sequence_length:.1f}",
                        str(stat.num_threads),
                        "Yes" if stat.cuda_enabled else "No",
                        f"{stat.compute_time_mean:.3f}",
                        f"{stat.aps_mean:,.0f}",
                    ]
                )
            else:
                rows.append(
                    [
                        stat.dataset,
                        stat.method.upper(),
                        stat.mode,
                        f"{stat.num_sequences:,}",
                        f"{stat.num_alignments:,}",
                        f"{stat.avg_sequence_length:.1f}",
                        str(stat.num_threads),
                        "Yes" if stat.cuda_enabled else "No",
                        f"{stat.compute_time_mean:.3f}",
                        f"±{stat.compute_time_std:.3f}",
                        f"{stat.aps_mean:,.0f}",
                        f"±{stat.aps_std:,.0f}",
                        f"{stat.coefficient_of_variation:.1f}%",
                        f"{stat.num_successful_runs}/{NUM_RUNS}",
                    ]
                )

    if is_single_run():
        align_values = [
            "left",
            "center",
            "center",
            "right",
            "right",
            "right",
            "center",
            "center",
            "right",
            "right",
        ]
    else:
        align_values = [
            "left",
            "center",
            "center",
            "right",
            "right",
            "right",
            "center",
            "center",
            "right",
            "right",
            "right",
            "right",
            "right",
            "center",
        ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="lightblue",
                    align="center",
                    font=dict(size=11, color="black"),
                    height=40,
                ),
                cells=dict(
                    values=list(zip(*rows)) if rows else [[] for _ in headers],
                    fill_color="white",
                    align=align_values,
                    font=dict(size=10),
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        title="Comprehensive Benchmark Results Summary",
        title_x=0.5,
        height=min(800, 100 + len(rows) * 35),
    )

    return fig


def main():
    in_project_root()

    print("Sequence Aligner Benchmark")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  {'Run' if is_single_run() else 'Runs'} per benchmark: {NUM_RUNS}")
    print(f"  Input files: {len(INPUT_FILES)}")
    print(f"  Algorithms: {ALIGNMENT_METHODS}")
    print(f"  Execution modes: {list(EXECUTION_MODES.keys())}")
    if not is_single_run():
        print(f"  Baseline mode: {BASELINE_MODE}")
    print(
        f"  Total configurations: {len(INPUT_FILES) * len(ALIGNMENT_METHODS) * len(EXECUTION_MODES)}"
    )
    print()

    manager = BenchmarkManager()

    for input_file in INPUT_FILES:
        if not Path(input_file).exists():
            print(f"✗ Input file not found: {input_file}")
            continue

        dataset_name = extract_dataset_name(input_file)
        manager.start_dataset(dataset_name)

        for method in ALIGNMENT_METHODS:
            for mode_name, mode_args in EXECUTION_MODES.items():
                config_num = len(manager.all_stats) + 1
                total_configs = (
                    len(INPUT_FILES) * len(ALIGNMENT_METHODS) * len(EXECUTION_MODES)
                )

                print(
                    f"[{config_num}/{total_configs}] {dataset_name}|{method}|{mode_name}"
                )

                runs = run_benchmarks(input_file, method, mode_name, mode_args)
                stats = calculate_statistics(runs)
                manager.add_benchmark_result(runs, stats)

                if stats.num_successful_runs > 0:
                    print(
                        f"  ✓ {stats.num_successful_runs}/{NUM_RUNS} successful {'run' if is_single_run() else 'runs'}"
                    )

                    if is_single_run():
                        print(f"    Compute time: {stats.compute_time_mean:.3f}s")
                        print(f"    Throughput: {stats.aps_mean:,.0f} APS")
                    else:
                        print(
                            f"    Mean compute time: {stats.compute_time_mean:.3f}±{stats.compute_time_std:.3f}s"
                        )
                        print(
                            f"    Mean throughput: {stats.aps_mean:,.0f}±{stats.aps_std:,.0f} APS"
                        )
                        print(
                            f"    Coefficient of variation: {stats.coefficient_of_variation:.1f}%"
                        )

                    print(f"    📄 Reports updated")
                else:
                    print(f"  ✗ All runs failed")

                print()

    print("Benchmark completed!")

    successful_configs = len(
        [s for s in manager.all_stats if s.num_successful_runs > 0]
    )
    total_runs = len(manager.all_runs)
    successful_runs = len([r for r in manager.all_runs if r.success])

    print(f"Final Summary:")
    print(f"  Successful configurations: {successful_configs}/{len(manager.all_stats)}")
    print(f"  Successful runs: {successful_runs}/{total_runs}")
    print(f"  Overall success rate: {successful_runs/total_runs*100:.1f}%")
    print()
    print(f"Generated reports:")
    print(f"  📁 {manager.results_dir}/")
    print(f"    📄 benchmark_combined_{TIMESTAMP}.* (all datasets)")

    datasets_processed = set(s.dataset for s in manager.all_stats)
    for dataset in sorted(datasets_processed):
        print(f"    📄 benchmark_{dataset}_{TIMESTAMP}.* (individual dataset)")


if __name__ == "__main__":
    main()
