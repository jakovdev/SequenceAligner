#!/usr/bin/env python3
import os
import sys
import re
import time
import json
import asyncio
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.colors import qualitative
from nicegui import ui

BINARY = "bin/seqalign"
CONFIG_FILE = "benchmark_config.json"
RESULTS_DIR = "benchmark_results"

AMINO_MATRICES = [
    'BLOSUM30', 'BLOSUM35', 'BLOSUM40', 'BLOSUM45', 'BLOSUM50', 'BLOSUM55',
    'BLOSUM60', 'BLOSUM62', 'BLOSUM65', 'BLOSUM70', 'BLOSUM75', 'BLOSUM80',
    'BLOSUM85', 'BLOSUM90', 'BLOSUM100',
    'PAM10', 'PAM20', 'PAM30', 'PAM40', 'PAM50', 'PAM60', 'PAM70', 'PAM80',
    'PAM90', 'PAM100', 'PAM110', 'PAM120', 'PAM130', 'PAM140', 'PAM150',
    'PAM160', 'PAM170', 'PAM180', 'PAM190', 'PAM200', 'PAM210', 'PAM220',
    'PAM230', 'PAM240', 'PAM250', 'PAM260', 'PAM270', 'PAM280', 'PAM290',
    'PAM300', 'PAM310', 'PAM320', 'PAM330', 'PAM340', 'PAM350', 'PAM360',
    'PAM370', 'PAM380', 'PAM390', 'PAM400', 'PAM410', 'PAM420', 'PAM430',
    'PAM440', 'PAM450', 'PAM460', 'PAM470', 'PAM480', 'PAM490', 'PAM500'
]
NUCLEO_MATRICES = ['DNAFULL', 'NUC44']
ALL_MATRICES = AMINO_MATRICES + NUCLEO_MATRICES


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
    alignments_per_second: float
    num_sequences: int
    num_alignments: int
    num_threads: int
    cuda_enabled: bool
    success: bool
    error_message: str = ""


@dataclass
class BenchmarkStats:
    dataset: str
    method: str
    profile_name: str
    compute_time_mean: float
    compute_time_std: float
    aps_mean: float
    normalized_throughput: float
    num_sequences: int
    num_alignments: int
    num_threads: int
    cuda_enabled: bool
    num_successful_runs: int
    speedup: Optional[float] = None


class BenchmarkEngine:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.binary = self._get_binary_path()
        
    def _get_binary_path(self) -> str:
        for suffix in ["", ".exe"]:
            path = Path(BINARY).with_suffix(suffix) if suffix else Path(BINARY)
            if path.exists():
                return str(path)
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
        
        for key, value in self.config.gap_penalties[method].items():
            cmd.extend([f"-{key}", str(value)])
        
        if not profile.is_cuda:
            cmd.extend(["-T", str(profile.threads), "-C"])
        
        return cmd
    
    def _parse_output(self, output: str) -> Dict[str, Any]:
        patterns = {
            "compute_time": (r"Compute:\s*([\d.]+)\s*sec", float, 0.0),
            "alignments_per_second": (r"Alignments per second:\s*([\d.]+)", float, 0.0),
            "num_sequences": (r"Found (\d+) sequences", int, 0),
            "num_alignments": (r"(?:Performing|Will perform) (\d+) pairwise alignments", int, 0),
            "num_threads": (r"(?:CPU Threads|Threads):\s*(\d+)", int, 0),
        }
        
        result = {}
        for key, (pattern, converter, default) in patterns.items():
            match = re.search(pattern, output)
            result[key] = converter(match.group(1)) if match else default
        
        cuda_match = re.search(r"CUDA:\s*(Enabled|Disabled)", output)
        result["cuda_enabled"] = cuda_match.group(1) == "Enabled" if cuda_match else False
        
        return result
    
    async def run_single(self, input_file: str, method: str, profile: ExecutionProfile, run_number: int, ui_ref=None) -> tuple[BenchmarkRun, str]:
        dataset_name = self._extract_dataset_name(input_file)
        cmd = self._build_command(input_file, method, profile)
        
        def create_failed_run(error_msg=""):
            return BenchmarkRun(
                dataset=dataset_name, method=method, profile_name=profile.name, run_number=run_number,
                compute_time=0, alignments_per_second=0, num_sequences=0, num_alignments=0,
                num_threads=0, cuda_enabled=False, success=False, error_message=error_msg
            )
        
        try:
            start_time = time.time()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            if ui_ref:
                ui_ref.current_process = process
            stdout, stderr = await process.communicate()
            if ui_ref:
                ui_ref.current_process = None
            end_time = time.time()
            
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            if process.returncode != 0:
                return create_failed_run(stderr_str), stderr_str
            
            parsed_data = self._parse_output(stdout_str)
            
            return BenchmarkRun(
                dataset=dataset_name,
                method=method,
                profile_name=profile.name,
                run_number=run_number,
                success=True,
                **parsed_data
            ), stdout_str
            
        except asyncio.TimeoutError:
            return create_failed_run("Timeout (3600s)"), "Timeout (3600s)"
        except Exception as e:
            return create_failed_run(str(e)), str(e)
    
    def calculate_statistics(self, runs: List[BenchmarkRun], 
                            baseline_stats: Optional['BenchmarkStats'] = None) -> Optional[BenchmarkStats]:
        df = pd.DataFrame([asdict(r) for r in runs if r.success])
        
        if df.empty:
            return None
        
        first_row = df.iloc[0]
        compute_mean = df['compute_time'].mean()
        normalized_throughput = first_row['num_alignments'] / compute_mean if compute_mean > 0 else 0
        speedup = baseline_stats.compute_time_mean / compute_mean if baseline_stats and compute_mean > 0 else None
        
        return BenchmarkStats(
            dataset=first_row['dataset'],
            method=first_row['method'],
            profile_name=first_row['profile_name'],
            compute_time_mean=compute_mean,
            compute_time_std=df['compute_time'].std() if len(df) > 1 else 0.0,
            aps_mean=df['alignments_per_second'].mean(),
            normalized_throughput=normalized_throughput,
            num_sequences=int(first_row['num_sequences']),
            num_alignments=int(first_row['num_alignments']),
            num_threads=int(first_row['num_threads']),
            cuda_enabled=bool(first_row['cuda_enabled']),
            num_successful_runs=len(df),
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
                "config": asdict(self.config),
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
        self.stop_requested = False
        self.current_process = None
        self.results_display = None
        self.results_ui_context = None
        self.log_drawer = None
        self.output_log = None
        self.progress_label = None
        self.progress_bar = None
        self.start_button = None
        self.stop_button = None
        
    def _load_config(self) -> BenchmarkConfig:
        if Path(CONFIG_FILE).exists():
            with open(CONFIG_FILE, 'r') as f:
                return BenchmarkConfig.from_dict(json.load(f))
        return BenchmarkConfig.default()
    
    def _save_config(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        ui.notify("Configuration saved.")
    
    def _add_subplot_data(self, fig, stats_list, datasets, profiles, method_colors, 
                          metric_extractor, row, col, show_legend=False, include_errors=False):
        for method in self.config.alignment_methods:
            for dataset in datasets:
                x_vals, y_vals, errors = [], [], []
                for profile in profiles:
                    stat = next((s for s in stats_list 
                               if s.dataset == dataset and s.method == method and s.profile_name == profile), None)
                    if stat:
                        value = metric_extractor(stat)
                        if value is not None:
                            x_vals.append(profile)
                            y_vals.append(value)
                            if include_errors:
                                errors.append(stat.compute_time_std)
                
                if x_vals:
                    trace = go.Bar(
                        x=x_vals, y=y_vals, name=f"{method.upper()}-{dataset}",
                        marker_color=method_colors[method], showlegend=show_legend
                    )
                    if include_errors and self.config.num_runs > 1:
                        trace.error_y = dict(type="data", array=errors)
                    fig.add_trace(trace, row=row, col=col)
    
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
        
        self._add_subplot_data(fig, stats_list, datasets, profiles, method_colors,
                              lambda s: s.compute_time_mean, 1, 1, show_legend=True, include_errors=True)
        self._add_subplot_data(fig, stats_list, datasets, profiles, method_colors,
                              lambda s: s.aps_mean, 1, 2)
        self._add_subplot_data(fig, stats_list, datasets, profiles, method_colors,
                              lambda s: s.normalized_throughput, 2, 1)
        self._add_subplot_data(fig, stats_list, datasets, profiles, method_colors,
                              lambda s: s.speedup, 2, 2)
        
        fig.update_layout(height=800, showlegend=True)
        fig.update_yaxes(title_text="Time (s)", type="log", row=1, col=1)
        fig.update_yaxes(title_text="APS", type="log", row=1, col=2)
        fig.update_yaxes(title_text="Pairs/s", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Speedup", row=2, col=2)
        
        return fig
    
    async def run_benchmark(self, progress_label, output_log):
        self.is_running = True
        self.stop_requested = False
        self.start_button.disable()
        self.stop_button.enable()
        self.engine = BenchmarkEngine(self.config)
        self.current_results = BenchmarkResults(self.config)
        
        total_configs = sum(
            len([m for m in self.config.alignment_methods if m in profile.methods])
            for profile in self.config.execution_profiles
        ) * len(self.config.input_files)
        
        current_config = 0
        baseline_stats_map = {}
        baseline_profile = self.config.get_baseline_profile()
        
        self.progress_bar.set_visibility(True)
        self.progress_bar.value = 0
        
        for input_file in self.config.input_files:
            dataset_name = self.engine._extract_dataset_name(input_file)
            
            for method in self.config.alignment_methods:
                for profile in self.config.execution_profiles:
                    if method not in profile.methods:
                        continue
                    
                    if self.stop_requested:
                        output_log.push("\n=== Benchmark stopped by user ===")
                        break
                    
                    current_config += 1
                    self.progress_bar.value = round(current_config / total_configs, 2)
                    
                    progress_label.text = f"Running {current_config}/{total_configs}: {dataset_name} | {method} | {profile.name}"
                    output_log.push(f"\n=== [{current_config}/{total_configs}] {dataset_name} | {method.upper()} | {profile.name} ===")
                    
                    runs = []
                    for i in range(self.config.num_runs):
                        if self.stop_requested:
                            output_log.push("  Stopping...")
                            break
                        
                        run, binary_output = await self.engine.run_single(input_file, method, profile, i + 1, self)
                        runs.append(run)
                        
                        if run.success:
                            output_log.push(f"  Run {i+1}: ✓ {run.compute_time:.3f}s ({run.alignments_per_second:,.0f} APS)")
                        else:
                            output_log.push(f"  Run {i+1}: ✗ {run.error_message[:100]}")
                        
                        await asyncio.sleep(0)
                    
                    baseline = None
                    if not profile.is_baseline and method in baseline_profile.methods:
                        baseline_key = (dataset_name, method, baseline_profile.name)
                        baseline = baseline_stats_map.get(baseline_key)
                    
                    stats = self.engine.calculate_statistics(runs, baseline)
                    
                    if stats:
                        output_log.push(f"  ✓ {stats.compute_time_mean:.3f}s ±{stats.compute_time_std:.3f}s")
                        if stats.speedup:
                            output_log.push(f"  Speedup: {stats.speedup:.2f}x")
                    
                    if stats and profile.is_baseline:
                        baseline_stats_map[(dataset_name, method, profile.name)] = stats
                    
                    self.current_results.add_result(runs, stats)
                    
                    if any(not r.success for r in runs) and self.log_drawer:
                        with self.results_ui_context:
                            self.log_drawer.show()
                    
                    if self.results_display and self.results_ui_context:
                        with self.results_ui_context:
                            self.results_display.refresh()
                
                if self.stop_requested:
                    break
            
            if self.stop_requested:
                break
        
        saved_file = self.current_results.save()
        status = "Stopped" if self.stop_requested else "Completed"
        progress_label.text = f"{status}. Results saved to {saved_file}"
        ui.notify(f"Benchmark {status.lower()}. Results saved to {saved_file.name}")
        
        self.is_running = False
        self.stop_requested = False
        self.progress_bar.set_visibility(False)
        self.start_button.enable()
        self.stop_button.disable()
    
    def create_config_ui(self, config_container):
        config_container.clear()
        
        with config_container:
            ui.label('Benchmark Configuration').classes('text-2xl mb-4')
            
            with ui.card().classes('w-full'):
                ui.label('General Settings & Gap Penalties').classes('text-xl mb-2')
                
                with ui.grid(columns=2).classes('w-full gap-6'):
                    with ui.column().classes('gap-2'):
                        ui.label('Run Configuration').classes('text-lg font-bold')
                        ui.number('Number of Runs', value=self.config.num_runs, 
                                 on_change=lambda e: setattr(self.config, 'num_runs', int(e.value)))
                    
                    with ui.column().classes('gap-2'):
                        ui.label('Gap Penalties').classes('text-lg font-bold')
                        
                        def update_penalty(method: str, key: str, value: int):
                            if value >= 1:
                                self.config.gap_penalties[method][key] = value
                            else:
                                ui.notify('Penalty must be positive', type='warning')
                        
                        with ui.row().classes('items-center gap-2'):
                            ui.label('NW gap penalty:').classes('w-28')
                            ui.number(value=self.config.gap_penalties['nw']['p'], min=1, step=1,
                                     on_change=lambda e: update_penalty('nw', 'p', int(e.value))).classes('w-12')
                        
                        with ui.row().classes('items-center gap-2'):
                            ui.label('GA gap open:').classes('w-28')
                            ui.number(value=self.config.gap_penalties['ga']['s'], min=1, step=1,
                                     on_change=lambda e: update_penalty('ga', 's', int(e.value))).classes('w-12')
                            ui.label('gap extend:').classes('w-20')
                            ui.number(value=self.config.gap_penalties['ga']['e'], min=1, step=1,
                                     on_change=lambda e: update_penalty('ga', 'e', int(e.value))).classes('w-12')
                        
                        with ui.row().classes('items-center gap-2'):
                            ui.label('SW gap open:').classes('w-28')
                            ui.number(value=self.config.gap_penalties['sw']['s'], min=1, step=1,
                                     on_change=lambda e: update_penalty('sw', 's', int(e.value))).classes('w-12')
                            ui.label('gap extend:').classes('w-20')
                            ui.number(value=self.config.gap_penalties['sw']['e'], min=1, step=1,
                                     on_change=lambda e: update_penalty('sw', 'e', int(e.value))).classes('w-12')
            
            with ui.grid(columns=2).classes('w-full gap-4 mt-4'):
                with ui.column().classes('gap-2'):
                    with ui.card().classes('w-full'):
                        ui.label('Execution Profiles').classes('text-xl')
                        
                        profiles_container = ui.column().classes('gap-2')
                        
                        def refresh_profiles():
                            profiles_container.clear()
                            with profiles_container:
                                for profile in self.config.execution_profiles:
                                    with ui.card().classes('w-full'):
                                        with ui.row().classes('justify-between items-center w-full'):
                                            ui.label(f"{profile.name}").classes('text-lg font-bold')
                                            ui.button('Delete', on_click=lambda p=profile: delete_profile(p)).props('size=sm flat color=negative')
                                        
                                        with ui.grid(columns=2).classes('w-full gap-1 text-sm'):
                                            ui.label(f"Type: {'CUDA' if profile.is_cuda else 'CPU'}")
                                            ui.label(f"Threads: {profile.threads}")
                                            ui.label(f"Baseline: {'Yes' if profile.is_baseline else 'No'}")
                                            ui.label(f"Methods: {', '.join(profile.methods).upper()}")
                        
                        def delete_profile(profile):
                            self.config.execution_profiles.remove(profile)
                            refresh_profiles()
                        
                        refresh_profiles()
                        
                        ui.separator()
                        ui.label('Add New Profile').classes('text-lg mt-2 font-bold')
                        profile_name = ui.input('Profile Name').classes('w-full')
                        with ui.grid(columns=3).classes('w-full gap-2'):
                            use_cuda = ui.checkbox('Use CUDA')
                            threads = ui.number('Threads', value=4, min=1, step=1).classes('w-24')
                            is_baseline = ui.checkbox('Is Baseline')
                        ui.label('Select Methods:').classes('text-sm mt-2')
                        with ui.row().classes('gap-4'):
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
                        
                        ui.button('Add Profile', on_click=add_profile).classes('mt-2')
                
                with ui.column().classes('gap-2'):
                    with ui.card().classes('w-full'):
                        ui.label('Input Files & Settings').classes('text-xl')
            
                        file_list_container = ui.column().classes('w-full gap-2')
                        
                        def refresh_file_list():
                            file_list_container.clear()
                            with file_list_container:
                                for file in self.config.input_files:
                                    dataset_name = Path(file).stem
                                    settings = self.config.file_settings.get(dataset_name, {})
                                    with ui.card().classes('w-full'):
                                        with ui.row().classes('justify-between items-center w-full'):
                                            ui.label(file).classes('text-sm font-bold')
                                            ui.button('Remove', on_click=lambda f=file: remove_file(f)).props('size=sm flat color=negative')
                                        if settings:
                                            ui.label(f"Type: {settings.get('sequence_type', 'N/A')} | Matrix: {settings.get('matrix', 'N/A')}").classes('text-xs')
                        
                        def remove_file(file):
                            self.config.input_files.remove(file)
                            dataset_name = Path(file).stem
                            if dataset_name in self.config.file_settings:
                                del self.config.file_settings[dataset_name]
                            refresh_file_list()
                        
                        refresh_file_list()
                        
                        ui.separator().classes('my-2')
                        ui.label('Add New File').classes('text-md font-bold')
                        new_file_input = ui.input('File path').classes('w-full')
                        seq_type_input = ui.select(['amino', 'nucleo'], label='Sequence type', value='amino').classes('w-full')
                        matrix_input = ui.select(ALL_MATRICES, label='Matrix', value='BLOSUM62').classes('w-full')
                        
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
                        
                        ui.button('Add File', on_click=add_file).classes('mt-2 w-full')
            
            ui.button('Save Configuration', on_click=lambda: self._save_config()).classes('mt-4 w-full').props('color=primary size=lg')
    
    def create_ui(self):
        @ui.page('/')
        async def main_page():
            with ui.header().classes('items-center justify-between'):
                ui.label('Sequence Aligner Benchmark').classes('text-3xl font-bold')
            
            if not Path(BINARY).exists() and not Path(f"{BINARY}.exe").exists():
                ui.banner('Warning: Binary not found at "bin/seqalign". Please build the project first.').props('type=warning')
            
            with ui.right_drawer(fixed=False).props('width=500') as log_drawer:
                self.log_drawer = log_drawer
                ui.label('Benchmark Output Log').classes('text-xl font-bold mb-2')
                self.output_log = ui.log().classes('w-full h-full')
            
            with ui.tabs().classes('w-full') as tabs:
                config_tab = ui.tab('Configuration')
                results_tab = ui.tab('Benchmark & Results')
                history_tab = ui.tab('History')
            
            with ui.tab_panels(tabs, value=config_tab).classes('w-full'):
                with ui.tab_panel(config_tab):
                    config_container = ui.column().classes('w-full')
                    self.create_config_ui(config_container)
                
                with ui.tab_panel(results_tab):
                    with ui.column().classes('w-full gap-2'):
                        with ui.row().classes('w-full items-center gap-4'):
                            self.progress_label = ui.label('Ready to run').classes('flex-grow text-lg')
                            
                            async def start_benchmark():
                                if not self.is_running:
                                    self.output_log.clear()
                                    self.log_drawer.show()
                                    await self.run_benchmark(self.progress_label, self.output_log)
                            
                            def stop_benchmark():
                                if self.is_running:
                                    self.stop_requested = True
                                    self.progress_label.text = "Stopping..."
                                    # Send SIGINT to current running process if it exists
                                    if self.current_process:
                                        try:
                                            self.current_process.send_signal(signal.SIGINT)
                                        except ProcessLookupError:
                                            pass  # Process already finished
                            
                            self.start_button = ui.button('Start Benchmark', on_click=start_benchmark, icon='play_arrow').props('color=primary')
                            self.stop_button = ui.button('Stop', on_click=stop_benchmark, icon='stop').props('color=negative')
                            self.stop_button.disable()
                        
                        self.progress_bar = ui.linear_progress(value=0).classes('w-full')
                        self.progress_bar.set_visibility(False)
                    
                    @ui.refreshable
                    def results_display():
                        if self.current_results and self.current_results.all_stats:
                            stats = self.current_results.all_stats
                            runs = self.current_results.all_runs
                            
                            ui.label(f"Benchmark Results - {self.current_results.timestamp}").classes('text-2xl font-bold mb-4')
                            
                            with ui.row().classes('gap-6 mb-4'):
                                with ui.card().classes('p-4'):
                                    ui.label('Configurations').classes('text-sm text-gray-600')
                                    ui.label(str(len(stats))).classes('text-2xl font-bold')
                                with ui.card().classes('p-4'):
                                    ui.label('Successful').classes('text-sm text-gray-600')
                                    ui.label(str(sum(1 for s in stats if s.num_successful_runs > 0))).classes('text-2xl font-bold text-green-600')
                                with ui.card().classes('p-4'):
                                    ui.label('Total Runs').classes('text-sm text-gray-600')
                                    ui.label(str(len(runs))).classes('text-2xl font-bold')
                                with ui.card().classes('p-4'):
                                    ui.label('Successful Runs').classes('text-sm text-gray-600')
                                    ui.label(str(sum(1 for r in runs if r.success))).classes('text-2xl font-bold text-green-600')
                            
                            ui.plotly(self.create_performance_plot(stats)).classes('w-full')
                            
                            with ui.row().classes('w-full items-center justify-between mt-6 mb-2'):
                                ui.label("Detailed Results").classes('text-xl font-bold')
                                
                                def export_csv():
                                    df = pd.DataFrame([asdict(s) for s in stats])
                                    csv_data = df.to_csv(index=False)
                                    ui.download(csv_data.encode(), f"benchmark_{self.current_results.timestamp}.csv")
                                    ui.notify('CSV exported successfully')
                                
                                ui.button('Export CSV', on_click=export_csv, icon='download').props('outline')
                            
                            columns = [
                                {'name': 'dataset', 'label': 'Dataset', 'field': 'dataset', 'align': 'left'},
                                {'name': 'method', 'label': 'Method', 'field': 'method', 'align': 'left'},
                                {'name': 'profile', 'label': 'Profile', 'field': 'profile', 'align': 'left'},
                                {'name': 'compute_time', 'label': 'Compute (s)', 'field': 'compute_time', 'align': 'right'},
                                {'name': 'aps', 'label': 'APS', 'field': 'aps', 'align': 'right'},
                                {'name': 'normalized', 'label': 'Norm. Throughput', 'field': 'normalized', 'align': 'right'},
                                {'name': 'speedup', 'label': 'Speedup', 'field': 'speedup', 'align': 'right'},
                            ]
                            
                            rows = [
                                {
                                    'dataset': stat.dataset,
                                    'method': stat.method.upper(),
                                    'profile': stat.profile_name,
                                    'compute_time': f"{stat.compute_time_mean:.3f} ±{stat.compute_time_std:.3f}",
                                    'aps': f"{stat.aps_mean:,.0f}",
                                    'normalized': f"{stat.normalized_throughput:.1f}",
                                    'speedup': f"{stat.speedup:.2f}x" if stat.speedup else "N/A",
                                }
                                for stat in stats
                            ]
                            
                            ui.table(columns=columns, rows=rows, row_key='dataset').classes('w-full')
                        else:
                            ui.label('No results available. Run a benchmark first.').classes('text-xl text-gray-600')
                    
                    self.results_display = results_display
                    self.results_ui_context = ui.context.client
                    results_display()
                
                with ui.tab_panel(history_tab):
                    ui.label('Previous Results').classes('text-2xl mb-4')
                    
                    results_dir = Path(RESULTS_DIR)
                    if results_dir.exists():
                        json_files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
                        
                        with ui.column().classes('gap-2 w-full'):
                            for json_file in json_files:
                                with ui.card().classes('w-full'):
                                    with ui.row().classes('justify-between items-center w-full'):
                                        with ui.column():
                                            ui.label(json_file.name).classes('font-bold')
                                            ui.label(f"Modified: {datetime.fromtimestamp(json_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}").classes('text-sm text-gray-600')
                                        
                                        def load_results(filepath=json_file):
                                            self.current_results = BenchmarkResults.load(str(filepath))
                                            ui.notify(f"Loaded {filepath.name}")
                                            if self.results_display:
                                                self.results_display.refresh()
                                            tabs.set_value(results_tab)
                                        
                                        ui.button('Load', on_click=load_results).props('color=primary')
                    else:
                        ui.label('No previous results found').classes('text-gray-600')
        
        ui.run(title='Sequence Aligner Benchmark', port=8080)


def main():
    os.chdir(Path.cwd() if Path.cwd().name != "script" else Path.cwd().parent)
    
    if not (Path("bin").exists() and Path("datasets").exists()):
        print("Error: Could not find project structure")
        sys.exit(1)
    
    app = BenchmarkUI()
    app.create_ui()


if __name__ in {"__main__", "__mp_main__"}:
    main()
