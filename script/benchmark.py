#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import time
import statistics
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.colors import qualitative
from nicegui import ui

BINARY = "bin/seqalign"
CONFIG_FILE = "benchmark_config.json"
RESULTS_DIR = "benchmark_results"


@dataclass
class ExecutionProfile:
    name: str
    is_cuda: bool
    threads: int
    is_baseline: bool
    methods: List[str]


@dataclass
class BenchmarkConfig:
    input_files: List[str]
    alignment_methods: List[str]
    execution_profiles: List[ExecutionProfile]
    num_runs: int
    file_settings: Dict[str, Dict[str, str]]
    gap_penalties: Dict[str, Dict[str, int]]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BenchmarkConfig':
        profiles = [ExecutionProfile(**p) for p in data['execution_profiles']]
        return BenchmarkConfig(
            input_files=data['input_files'],
            alignment_methods=data['alignment_methods'],
            execution_profiles=profiles,
            num_runs=data['num_runs'],
            file_settings=data['file_settings'],
            gap_penalties=data['gap_penalties']
        )
    
    @staticmethod
    def default() -> 'BenchmarkConfig':
        return BenchmarkConfig(
            input_files=["datasets/avppred.csv", "datasets/amp.csv"],
            alignment_methods=["nw", "ga", "sw"],
            execution_profiles=[
                ExecutionProfile("CPU-01T", False, 1, True, ["nw", "ga", "sw"]),
                ExecutionProfile("CPU-04T", False, 4, False, ["nw", "ga", "sw"]),
                ExecutionProfile("CPU-08T", False, 8, False, ["nw", "ga", "sw"]),
                ExecutionProfile("CPU-16T", False, 16, False, ["nw", "ga", "sw"]),
                ExecutionProfile("CUDA", True, 1024, False, ["nw", "ga", "sw"]),
            ],
            num_runs=3,
            file_settings={
                "avppred": {"sequence_type": "amino", "matrix": "BLOSUM62"},
                "amp": {"sequence_type": "amino", "matrix": "BLOSUM62"},
                "celegans": {"sequence_type": "amino", "matrix": "BLOSUM62"},
                "drosophila": {"sequence_type": "amino", "matrix": "BLOSUM62"},
                "mouse": {"sequence_type": "amino", "matrix": "BLOSUM62"},
            },
            gap_penalties={
                "nw": {"p": 4},
                "ga": {"s": 10, "e": 1},
                "sw": {"s": 10, "e": 1},
            }
        )
    
    def get_baseline_profile(self) -> ExecutionProfile:
        for profile in self.execution_profiles:
            if profile.is_baseline:
                return profile
        return self.execution_profiles[0]


@dataclass
class BenchmarkRun:
    dataset: str
    method: str
    profile_name: str
    run_number: int
    compute_time: float
    total_time: float
    wall_time: float
    alignments_per_second: float
    alignments_per_second_per_thread: float
    avg_time_per_thread: float
    num_sequences: int
    num_alignments: int
    avg_sequence_length: float
    num_threads: int
    cuda_enabled: bool
    matrix_checksum: int
    success: bool
    error_message: str = ""


@dataclass
class BenchmarkStats:
    dataset: str
    method: str
    profile_name: str
    compute_time_mean: float
    compute_time_std: float
    compute_time_min: float
    compute_time_max: float
    total_time_mean: float
    total_time_std: float
    aps_mean: float
    aps_std: float
    aps_min: float
    aps_max: float
    normalized_time: float
    normalized_throughput: float
    num_sequences: int
    num_alignments: int
    avg_sequence_length: float
    num_threads: int
    cuda_enabled: bool
    num_successful_runs: int
    coefficient_of_variation: float
    speedup: Optional[float] = None


class BenchmarkEngine:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.binary = self._get_binary_path()
        
    def _get_binary_path(self) -> str:
        if Path(BINARY).exists():
            return BINARY
        if Path(f"{BINARY}.exe").exists():
            return f"{BINARY}.exe"
        raise FileNotFoundError(f"Binary not found: {BINARY}")
    
    def _extract_dataset_name(self, input_file: str) -> str:
        return Path(input_file).stem
    
    def _build_command(self, input_file: str, method: str, profile: ExecutionProfile) -> List[str]:
        dataset_name = self._extract_dataset_name(input_file)
        settings = self.config.file_settings.get(dataset_name, {"sequence_type": "amino", "matrix": "BLOSUM62"})
        
        cmd = [
            self.binary,
            "-i", input_file,
            "-t", settings["sequence_type"],
            "-m", settings["matrix"],
            "-a", method,
            "-DBW"
        ]
        
        penalties = self.config.gap_penalties[method]
        for key, value in penalties.items():
            cmd.extend([f"-{key}", str(value)])
        
        if profile.is_cuda:
            pass
        else:
            cmd.extend(["-T", str(profile.threads), "-C"])
        
        return cmd
    
    def _parse_output(self, output: str) -> Dict[str, Any]:
        compute_match = re.search(r"Compute:\s*([\d.]+)\s*sec", output)
        total_match = re.search(r"Total:\s*([\d.]+)\s*sec", output)
        aps_match = re.search(r"Alignments per second:\s*([\d.]+)", output)
        aps_per_thread_match = re.search(r"Alignments per second per thread:\s*([\d.]+)", output)
        avg_time_match = re.search(r"Average time per thread:\s*([\d.]+)\s*sec", output)
        num_seq_match = re.search(r"Found (\d+) sequences", output)
        num_align_match = re.search(r"(?:Performing|Will perform) (\d+) pairwise alignments", output)
        avg_len_match = re.search(r"Average sequence length:\s*([\d.]+)", output)
        threads_match = re.search(r"(?:CPU Threads|Threads):\s*(\d+)", output)
        cuda_match = re.search(r"CUDA:\s*(Enabled|Disabled)", output)
        checksum_match = re.search(r"Matrix checksum:\s*(-?\d+)", output)
        
        return {
            "compute_time": float(compute_match.group(1)) if compute_match else 0.0,
            "total_time": float(total_match.group(1)) if total_match else 0.0,
            "alignments_per_second": float(aps_match.group(1)) if aps_match else 0.0,
            "alignments_per_second_per_thread": float(aps_per_thread_match.group(1)) if aps_per_thread_match else 0.0,
            "avg_time_per_thread": float(avg_time_match.group(1)) if avg_time_match else 0.0,
            "num_sequences": int(num_seq_match.group(1)) if num_seq_match else 0,
            "num_alignments": int(num_align_match.group(1)) if num_align_match else 0,
            "avg_sequence_length": float(avg_len_match.group(1)) if avg_len_match else 0.0,
            "num_threads": int(threads_match.group(1)) if threads_match else 0,
            "cuda_enabled": cuda_match.group(1) == "Enabled" if cuda_match else False,
            "matrix_checksum": int(checksum_match.group(1)) if checksum_match else 0
        }
    
    async def run_single(self, input_file: str, method: str, profile: ExecutionProfile, run_number: int) -> tuple[BenchmarkRun, str]:
        dataset_name = self._extract_dataset_name(input_file)
        cmd = self._build_command(input_file, method, profile)
        
        try:
            start_time = time.time()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            end_time = time.time()
            
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            if process.returncode != 0:
                return BenchmarkRun(
                    dataset=dataset_name, method=method, profile_name=profile.name, run_number=run_number,
                    compute_time=0, total_time=0, wall_time=end_time - start_time,
                    alignments_per_second=0, alignments_per_second_per_thread=0,
                    avg_time_per_thread=0, num_sequences=0, num_alignments=0,
                    avg_sequence_length=0, num_threads=0, cuda_enabled=False,
                    matrix_checksum=0, success=False, error_message=stderr_str
                ), stderr_str
            
            parsed_data = self._parse_output(stdout_str)
            
            return BenchmarkRun(
                dataset=dataset_name,
                method=method,
                profile_name=profile.name,
                run_number=run_number,
                wall_time=end_time - start_time,
                success=True,
                **parsed_data
            ), stdout_str
            
        except asyncio.TimeoutError:
            return BenchmarkRun(
                dataset=dataset_name, method=method, profile_name=profile.name, run_number=run_number,
                compute_time=0, total_time=0, wall_time=3600,
                alignments_per_second=0, alignments_per_second_per_thread=0,
                avg_time_per_thread=0, num_sequences=0, num_alignments=0,
                avg_sequence_length=0, num_threads=0, cuda_enabled=False,
                matrix_checksum=0, success=False, error_message="Timeout (3600s)"
            ), "Timeout (3600s)"
        except Exception as e:
            return BenchmarkRun(
                dataset=dataset_name, method=method, profile_name=profile.name, run_number=run_number,
                compute_time=0, total_time=0, wall_time=0,
                alignments_per_second=0, alignments_per_second_per_thread=0,
                avg_time_per_thread=0, num_sequences=0, num_alignments=0,
                avg_sequence_length=0, num_threads=0, cuda_enabled=False,
                matrix_checksum=0, success=False, error_message=str(e)
            ), str(e)
    
    def calculate_statistics(self, runs: List[BenchmarkRun], 
                            baseline_stats: Optional['BenchmarkStats'] = None) -> Optional[BenchmarkStats]:
        successful_runs = [r for r in runs if r.success]
        
        if not successful_runs:
            return None
        
        compute_times = [r.compute_time for r in successful_runs]
        total_times = [r.total_time for r in successful_runs]
        aps_values = [r.alignments_per_second for r in successful_runs]
        
        first_run = successful_runs[0]
        
        n = first_run.num_sequences
        alignment_pairs = n * (n - 1) // 2
        normalized_time = statistics.mean(compute_times) / alignment_pairs if alignment_pairs > 0 else 0
        normalized_throughput = alignment_pairs / statistics.mean(compute_times) if statistics.mean(compute_times) > 0 else 0
        
        speedup = None
        if baseline_stats and baseline_stats.compute_time_mean > 0 and statistics.mean(compute_times) > 0:
            speedup = baseline_stats.compute_time_mean / statistics.mean(compute_times)
        
        return BenchmarkStats(
            dataset=first_run.dataset,
            method=first_run.method,
            profile_name=first_run.profile_name,
            compute_time_mean=statistics.mean(compute_times),
            compute_time_std=statistics.stdev(compute_times) if len(compute_times) > 1 else 0.0,
            compute_time_min=min(compute_times),
            compute_time_max=max(compute_times),
            total_time_mean=statistics.mean(total_times),
            total_time_std=statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
            aps_mean=statistics.mean(aps_values),
            aps_std=statistics.stdev(aps_values) if len(aps_values) > 1 else 0.0,
            aps_min=min(aps_values),
            aps_max=max(aps_values),
            normalized_time=normalized_time,
            normalized_throughput=normalized_throughput,
            num_sequences=first_run.num_sequences,
            num_alignments=first_run.num_alignments,
            avg_sequence_length=first_run.avg_sequence_length,
            num_threads=first_run.num_threads,
            cuda_enabled=first_run.cuda_enabled,
            num_successful_runs=len(successful_runs),
            coefficient_of_variation=(
                (statistics.stdev(compute_times) / statistics.mean(compute_times) * 100)
                if len(compute_times) > 1 else 0.0
            ),
            speedup=speedup
        )


class BenchmarkResults:
    def __init__(self, config: BenchmarkConfig, timestamp: str = None):
        self.config = config
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_runs: List[BenchmarkRun] = []
        self.all_stats: List[BenchmarkStats] = []
        
    def add_result(self, runs: List[BenchmarkRun], stats: Optional[BenchmarkStats]):
        self.all_runs.extend(runs)
        if stats:
            self.all_stats.append(stats)
    
    def save(self):
        results_dir = Path(RESULTS_DIR)
        results_dir.mkdir(exist_ok=True)
        
        data = {
            "metadata": {
                "timestamp": self.timestamp,
                "config": self.config.to_dict(),
                "total_configurations": len(self.all_stats),
                "successful_configurations": len([s for s in self.all_stats if s.num_successful_runs > 0]),
                "total_runs": len(self.all_runs),
                "successful_runs": len([r for r in self.all_runs if r.success]),
            },
            "statistics": [asdict(stat) for stat in self.all_stats],
            "raw_runs": [asdict(run) for run in self.all_runs],
        }
        
        json_file = results_dir / f"benchmark_{self.timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        return json_file
    
    @staticmethod
    def load(filepath: str) -> 'BenchmarkResults':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = BenchmarkConfig.from_dict(data["metadata"]["config"])
        timestamp = data["metadata"]["timestamp"]
        
        results = BenchmarkResults(config, timestamp)
        results.all_stats = [BenchmarkStats(**s) for s in data["statistics"]]
        results.all_runs = [BenchmarkRun(**r) for r in data["raw_runs"]]
        
        return results


class BenchmarkUI:
    def __init__(self):
        self.config = self._load_config()
        self.current_results: Optional[BenchmarkResults] = None
        self.engine: Optional[BenchmarkEngine] = None
        self.is_running = False
        
    def _load_config(self) -> BenchmarkConfig:
        if Path(CONFIG_FILE).exists():
            with open(CONFIG_FILE, 'r') as f:
                return BenchmarkConfig.from_dict(json.load(f))
        return BenchmarkConfig.default()
    
    def _save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        ui.notify("Configuration saved.")
    
    def create_performance_plot(self, stats_list: List[BenchmarkStats]) -> go.Figure:
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Compute Time by Algorithm & Profile",
                "Throughput (Alignments/sec)",
                "Normalized Performance (n*(n-1)/2)",
                "Speedup vs Baseline"
            ),
        )
        
        colors = qualitative.Set1
        method_colors = {method: colors[i % len(colors)] 
                        for i, method in enumerate(self.config.alignment_methods)}
        
        datasets = sorted(set(s.dataset for s in stats_list))
        profiles = sorted(set(s.profile_name for s in stats_list))
        
        for method in self.config.alignment_methods:
            for dataset in datasets:
                x_vals, y_vals, errors = [], [], []
                for profile in profiles:
                    stat = next((s for s in stats_list 
                               if s.dataset == dataset and s.method == method and s.profile_name == profile), None)
                    if stat:
                        x_vals.append(profile)
                        y_vals.append(stat.compute_time_mean)
                        errors.append(stat.compute_time_std)
                
                if x_vals:
                    fig.add_trace(go.Bar(
                        x=x_vals, y=y_vals, name=f"{method.upper()}-{dataset}",
                        marker_color=method_colors[method],
                        error_y=dict(type="data", array=errors) if self.config.num_runs > 1 else None
                    ), row=1, col=1)
        
        for method in self.config.alignment_methods:
            for dataset in datasets:
                x_vals, y_vals = [], []
                for profile in profiles:
                    stat = next((s for s in stats_list 
                               if s.dataset == dataset and s.method == method and s.profile_name == profile), None)
                    if stat:
                        x_vals.append(profile)
                        y_vals.append(stat.aps_mean)
                
                if x_vals:
                    fig.add_trace(go.Bar(
                        x=x_vals, y=y_vals, name=f"{method.upper()}-{dataset}",
                        marker_color=method_colors[method], showlegend=False
                    ), row=1, col=2)
        
        for method in self.config.alignment_methods:
            for dataset in datasets:
                x_vals, y_vals = [], []
                for profile in profiles:
                    stat = next((s for s in stats_list 
                               if s.dataset == dataset and s.method == method and s.profile_name == profile), None)
                    if stat:
                        x_vals.append(profile)
                        y_vals.append(stat.normalized_throughput)
                
                if x_vals:
                    fig.add_trace(go.Bar(
                        x=x_vals, y=y_vals, name=f"{method.upper()}-{dataset}",
                        marker_color=method_colors[method], showlegend=False
                    ), row=2, col=1)
        
        for method in self.config.alignment_methods:
            for dataset in datasets:
                x_vals, y_vals = [], []
                for profile in profiles:
                    stat = next((s for s in stats_list 
                               if s.dataset == dataset and s.method == method and s.profile_name == profile), None)
                    if stat and stat.speedup:
                        x_vals.append(profile)
                        y_vals.append(stat.speedup)
                
                if x_vals:
                    fig.add_trace(go.Bar(
                        x=x_vals, y=y_vals, name=f"{method.upper()}-{dataset}",
                        marker_color=method_colors[method], showlegend=False
                    ), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True)
        fig.update_yaxes(title_text="Time (s)", type="log", row=1, col=1)
        fig.update_yaxes(title_text="APS", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Pairs/s", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Speedup", row=2, col=2)
        
        return fig
    
    async def run_benchmark(self, progress_label, output_log):
        self.is_running = True
        self.engine = BenchmarkEngine(self.config)
        self.current_results = BenchmarkResults(self.config)
        
        total_configs = sum(
            len([m for m in self.config.alignment_methods if m in profile.methods])
            for profile in self.config.execution_profiles
        ) * len(self.config.input_files)
        
        current_config = 0
        baseline_stats_map = {}
        baseline_profile = self.config.get_baseline_profile()
        
        for input_file in self.config.input_files:
            dataset_name = self.engine._extract_dataset_name(input_file)
            
            for method in self.config.alignment_methods:
                for profile in self.config.execution_profiles:
                    if method not in profile.methods:
                        continue
                    
                    current_config += 1
                    
                    progress_label.text = f"Running {current_config}/{total_configs}: {dataset_name} | {method} | {profile.name}"
                    
                    try:
                        output_log.push(f"\n=== [{current_config}/{total_configs}] {dataset_name} | {method.upper()} | {profile.name} ===")
                    except:
                        pass
                    
                    runs = []
                    for i in range(self.config.num_runs):
                        run, binary_output = await self.engine.run_single(input_file, method, profile, i + 1)
                        runs.append(run)
                        
                        try:
                            if run.success:
                                output_log.push(f"  Run {i+1}: ✓ {run.compute_time:.3f}s ({run.alignments_per_second:,.0f} APS)")
                            else:
                                output_log.push(f"  Run {i+1}: ✗ {run.error_message[:100]}")
                        except:
                            pass
                        
                        await asyncio.sleep(0)
                    
                    baseline = None
                    if not profile.is_baseline and method in baseline_profile.methods:
                        baseline_key = (dataset_name, method, baseline_profile.name)
                        baseline = baseline_stats_map.get(baseline_key)
                    
                    stats = self.engine.calculate_statistics(runs, baseline)
                    
                    try:
                        if stats:
                            output_log.push(f"  ✓ {stats.compute_time_mean:.3f}s ±{stats.compute_time_std:.3f}s")
                            if stats.speedup:
                                output_log.push(f"  Speedup: {stats.speedup:.2f}x")
                    except:
                        pass
                    
                    if stats and profile.is_baseline:
                        baseline_stats_map[(dataset_name, method, profile.name)] = stats
                    
                    self.current_results.add_result(runs, stats)
        
        saved_file = self.current_results.save()
        progress_label.text = f"Completed. Results saved to {saved_file}"
        ui.notify(f"Benchmark complete. Results saved to {saved_file.name}")
        
        self.is_running = False
    
    def display_results(self, container):
        container.clear()
        
        if not self.current_results or not self.current_results.all_stats:
            with container:
                ui.label("No results to display")
            return
        
        with container:
            ui.label(f"Benchmark Results - {self.current_results.timestamp}").classes('text-2xl font-bold')
            
            with ui.row():
                ui.label(f"Configurations: {len(self.current_results.all_stats)}")
                ui.label(f"Successful: {len([s for s in self.current_results.all_stats if s.num_successful_runs > 0])}")
                ui.label(f"Total Runs: {len(self.current_results.all_runs)}")
                ui.label(f"Successful Runs: {len([r for r in self.current_results.all_runs if r.success])}")
            
            fig = self.create_performance_plot(self.current_results.all_stats)
            ui.plotly(fig).classes('w-full')
            
            ui.label("Detailed Results").classes('text-xl font-bold mt-4')
            
            columns = [
                {'name': 'dataset', 'label': 'Dataset', 'field': 'dataset', 'align': 'left'},
                {'name': 'method', 'label': 'Method', 'field': 'method', 'align': 'left'},
                {'name': 'profile', 'label': 'Profile', 'field': 'profile', 'align': 'left'},
                {'name': 'compute_time', 'label': 'Compute (s)', 'field': 'compute_time', 'align': 'right'},
                {'name': 'aps', 'label': 'APS', 'field': 'aps', 'align': 'right'},
                {'name': 'normalized', 'label': 'Norm. Throughput', 'field': 'normalized', 'align': 'right'},
                {'name': 'speedup', 'label': 'Speedup', 'field': 'speedup', 'align': 'right'},
                {'name': 'cv', 'label': 'CV%', 'field': 'cv', 'align': 'right'},
            ]
            
            rows = []
            for stat in self.current_results.all_stats:
                rows.append({
                    'dataset': stat.dataset,
                    'method': stat.method.upper(),
                    'profile': stat.profile_name,
                    'compute_time': f"{stat.compute_time_mean:.3f} ±{stat.compute_time_std:.3f}",
                    'aps': f"{stat.aps_mean:,.0f}",
                    'normalized': f"{stat.normalized_throughput:.1f}",
                    'speedup': f"{stat.speedup:.2f}x" if stat.speedup else "N/A",
                    'cv': f"{stat.coefficient_of_variation:.1f}%",
                })
            
            ui.table(columns=columns, rows=rows, row_key='dataset').classes('w-full')
    
    def create_config_ui(self, config_container):
        config_container.clear()
        
        with config_container:
            ui.label('Benchmark Configuration').classes('text-2xl')
            
            with ui.card():
                ui.label('General Settings').classes('text-xl')
                ui.number('Number of Runs', value=self.config.num_runs, 
                         on_change=lambda e: setattr(self.config, 'num_runs', int(e.value)))
            
            with ui.card():
                ui.label('Input Files & Settings').classes('text-xl')
                
                file_list_container = ui.column()
                
                amino_matrices = [
                    'BLOSUM100', 'BLOSUM30', 'BLOSUM35', 'BLOSUM40', 'BLOSUM45',
                    'BLOSUM50', 'BLOSUM55', 'BLOSUM60', 'BLOSUM62', 'BLOSUM65',
                    'BLOSUM70', 'BLOSUM75', 'BLOSUM80', 'BLOSUM85', 'BLOSUM90',
                    'PAM10', 'PAM100', 'PAM110', 'PAM120', 'PAM130',
                    'PAM140', 'PAM150', 'PAM160', 'PAM170', 'PAM180',
                    'PAM190', 'PAM20', 'PAM200', 'PAM210', 'PAM220',
                    'PAM230', 'PAM240', 'PAM250', 'PAM260', 'PAM270',
                    'PAM280', 'PAM290', 'PAM30', 'PAM300', 'PAM310',
                    'PAM320', 'PAM330', 'PAM340', 'PAM350', 'PAM360',
                    'PAM370', 'PAM380', 'PAM390', 'PAM40', 'PAM400',
                    'PAM410', 'PAM420', 'PAM430', 'PAM440', 'PAM450',
                    'PAM460', 'PAM470', 'PAM480', 'PAM490', 'PAM50',
                    'PAM500', 'PAM60', 'PAM70', 'PAM80', 'PAM90'
                ]
                nucleo_matrices = ['DNAFULL', 'NUC44']
                all_matrices = amino_matrices + nucleo_matrices
                
                def refresh_file_list():
                    file_list_container.clear()
                    with file_list_container:
                        for file in self.config.input_files:
                            dataset_name = Path(file).stem
                            settings = self.config.file_settings.get(dataset_name, {})
                            with ui.card().classes('w-full'):
                                with ui.row():
                                    ui.label(file).classes('text-md font-bold')
                                    ui.button('Remove', on_click=lambda f=file: remove_file(f)).props('size=sm flat color=negative')
                                if settings:
                                    ui.label(f"Type: {settings.get('sequence_type', 'N/A')} | Matrix: {settings.get('matrix', 'N/A')}")
                
                def remove_file(file):
                    self.config.input_files.remove(file)
                    dataset_name = Path(file).stem
                    if dataset_name in self.config.file_settings:
                        del self.config.file_settings[dataset_name]
                    refresh_file_list()
                
                refresh_file_list()
                
                ui.label('Add New File').classes('text-lg mt-4')
                with ui.row():
                    new_file_input = ui.input('File path')
                    seq_type_input = ui.select(['amino', 'nucleo'], label='Sequence type', value='amino')
                    matrix_input = ui.select(all_matrices, label='Matrix', value='BLOSUM62')
                
                def add_file():
                    path = new_file_input.value
                    if path and path not in self.config.input_files:
                        self.config.input_files.append(path)
                        dataset_name = Path(path).stem
                        self.config.file_settings[dataset_name] = {
                            'sequence_type': seq_type_input.value,
                            'matrix': matrix_input.value
                        }
                        refresh_file_list()
                        new_file_input.value = ''
                        seq_type_input.value = 'amino'
                        matrix_input.value = 'BLOSUM62'
                
                ui.button('Add File', on_click=add_file)
            
            with ui.card():
                ui.label('Execution Profiles').classes('text-xl')
                
                profiles_container = ui.column()
                
                def refresh_profiles():
                    profiles_container.clear()
                    with profiles_container:
                        for profile in self.config.execution_profiles:
                            with ui.card().classes('w-full'):
                                with ui.row():
                                    ui.label(f"{profile.name}").classes('text-lg font-bold')
                                    ui.button('Delete', on_click=lambda p=profile: delete_profile(p)).props('size=sm flat color=negative')
                                
                                ui.label(f"Type: {'CUDA' if profile.is_cuda else 'CPU'}")
                                ui.label(f"Threads: {profile.threads}")
                                ui.label(f"Baseline: {'Yes' if profile.is_baseline else 'No'}")
                                ui.label(f"Methods: {', '.join(profile.methods).upper()}")
                
                def delete_profile(profile):
                    self.config.execution_profiles.remove(profile)
                    refresh_profiles()
                
                refresh_profiles()
                
                ui.label('Add New Profile').classes('text-lg mt-4')
                profile_name = ui.input('Profile Name')
                with ui.row():
                    use_cuda = ui.checkbox('Use CUDA')
                    threads = ui.number('Threads', value=4, min=1, step=1)
                    is_baseline = ui.checkbox('Is Baseline')
                ui.label('Select Methods:')
                with ui.row():
                    method_nw = ui.checkbox('NW', value=True)
                    method_ga = ui.checkbox('GA', value=True)
                    method_sw = ui.checkbox('SW', value=True)
                
                def add_profile():
                    if profile_name.value:
                        selected_methods = []
                        if method_nw.value:
                            selected_methods.append('nw')
                        if method_ga.value:
                            selected_methods.append('ga')
                        if method_sw.value:
                            selected_methods.append('sw')
                        
                        if not selected_methods:
                            ui.notify('Please select at least one method', type='warning')
                            return
                        
                        new_profile = ExecutionProfile(
                            name=profile_name.value,
                            is_cuda=use_cuda.value,
                            threads=int(threads.value),
                            is_baseline=is_baseline.value,
                            methods=selected_methods
                        )
                        self.config.execution_profiles.append(new_profile)
                        refresh_profiles()
                        profile_name.value = ''
                        threads.value = 4
                        use_cuda.value = False
                        is_baseline.value = False
                        method_nw.value = True
                        method_ga.value = True
                        method_sw.value = True
                
                ui.button('Add Profile', on_click=add_profile)
            
            with ui.card():
                ui.label('Gap Penalties').classes('text-xl')
                
                gaps_container = ui.column()
                
                def refresh_gaps():
                    gaps_container.clear()
                    with gaps_container:
                        ui.label('NW (Needleman-Wunsch)').classes('font-bold')
                        with ui.row():
                            ui.label('-p:')
                            nw_p = ui.number('Penalty', value=self.config.gap_penalties['nw']['p'], min=1, step=1)
                            nw_p.on('change', lambda e: update_penalty('nw', 'p', int(e.value)))
                        
                        ui.label('GA (Global Affine)').classes('font-bold mt-2')
                        with ui.row():
                            ui.label('-s:')
                            ga_s = ui.number('Start', value=self.config.gap_penalties['ga']['s'], min=1, step=1)
                            ga_s.on('change', lambda e: update_penalty('ga', 's', int(e.value)))
                            ui.label('-e:')
                            ga_e = ui.number('Extend', value=self.config.gap_penalties['ga']['e'], min=1, step=1)
                            ga_e.on('change', lambda e: update_penalty('ga', 'e', int(e.value)))
                        
                        ui.label('SW (Smith-Waterman)').classes('font-bold mt-2')
                        with ui.row():
                            ui.label('-s:')
                            sw_s = ui.number('Start', value=self.config.gap_penalties['sw']['s'], min=1, step=1)
                            sw_s.on('change', lambda e: update_penalty('sw', 's', int(e.value)))
                            ui.label('-e:')
                            sw_e = ui.number('Extend', value=self.config.gap_penalties['sw']['e'], min=1, step=1)
                            sw_e.on('change', lambda e: update_penalty('sw', 'e', int(e.value)))
                
                def update_penalty(method: str, key: str, value: int):
                    if value < 1:
                        ui.notify('Penalty must be positive', type='warning')
                        return
                    self.config.gap_penalties[method][key] = value
                
                refresh_gaps()
            
            ui.button('Save Configuration', on_click=lambda: self._save_config())
    
    def create_ui(self):
        @ui.page('/')
        async def main_page():
            ui.label('Sequence Aligner Benchmark').classes('text-3xl font-bold')
            
            with ui.tabs() as tabs:
                config_tab = ui.tab('Configuration')
                run_tab = ui.tab('Run Benchmark')
                results_tab = ui.tab('Results')
                history_tab = ui.tab('History')
            
            with ui.tab_panels(tabs, value=config_tab):
                with ui.tab_panel(config_tab):
                    config_container = ui.column()
                    self.create_config_ui(config_container)
                
                with ui.tab_panel(run_tab):
                    ui.label('Run Benchmark').classes('text-2xl')
                    
                    progress_label = ui.label('Ready to run')
                    output_log = ui.log().classes('w-full h-96')
                    
                    async def start_benchmark():
                        if not self.is_running:
                            output_log.clear()
                            await self.run_benchmark(progress_label, output_log)
                    
                    ui.button('Start Benchmark', on_click=start_benchmark)
                
                with ui.tab_panel(results_tab):
                    ui.label('Current Results').classes('text-2xl')
                    current_results_container = ui.column()
                    
                    def refresh_results():
                        if self.current_results:
                            self.display_results(current_results_container)
                        else:
                            current_results_container.clear()
                            with current_results_container:
                                ui.label('No results available. Run a benchmark first.')
                    
                    ui.button('Refresh Results', on_click=refresh_results)
                    
                    if self.current_results:
                        self.display_results(current_results_container)
                
                with ui.tab_panel(history_tab):
                    ui.label('Previous Results').classes('text-2xl')
                    
                    results_dir = Path(RESULTS_DIR)
                    if results_dir.exists():
                        json_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
                        
                        for json_file in json_files:
                            with ui.row():
                                ui.label(json_file.name)
                                
                                def load_results(filepath=json_file):
                                    self.current_results = BenchmarkResults.load(str(filepath))
                                    ui.notify(f"Loaded {filepath.name}")
                                    tabs.set_value(results_tab)
                                    self.display_results(current_results_container)
                                
                                ui.button('Load', on_click=load_results)
                    else:
                        ui.label('No previous results found')
        
        ui.run(title='Sequence Aligner Benchmark', port=8080)


def main():
    current_dir = Path.cwd()
    if current_dir.name == "script":
        os.chdir(current_dir.parent)
    
    if not (Path("bin").exists() and Path("datasets").exists()):
        print("Error: Could not find project structure")
        sys.exit(1)
    
    app = BenchmarkUI()
    app.create_ui()


if __name__ in {"__main__", "__mp_main__"}:
    main()
