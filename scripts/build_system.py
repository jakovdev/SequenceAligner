import platform
import threading
import subprocess
import atexit
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

class BuildEnvironment:
    def __init__(self):
        self.os_name = platform.system().lower()
        self.make_cmd = "mingw32-make" if self.os_name == "windows" else "make"
        self._lock = threading.Lock()
        self._build_states = {"all": False}
        self._active_processes = {}
        self._state_callbacks = set()
        self._is_building = False
        atexit.register(self._cleanup_processes)

    def verify_windows_environment(self, output_fn):
        import urllib.request
        import os
        import shutil
        import itertools
        import time

        MSYS2_PATHS = {
            "SCOOP_GLOBAL": "C:/ProgramData/scoop/apps/msys2/current/ucrt64/bin",
            "SCOOP_USER": str(Path.home() / "scoop/apps/msys2/current/ucrt64/bin"),
            "MSYS2_DEFAULT": "C:/msys64/ucrt64/bin",
            "MINGW64_PROJECT_LOCAL": str(project_root / "w64devkit/bin"),
        }

        def verify_gcc_make():
            try:
                for cmd in ["gcc --version", "mingw32-make --version"]:
                    result = subprocess.run(
                        cmd, shell=True, check=True, capture_output=True, timeout=2
                    )
                    if result.returncode != 0:
                        return False
                return True
            except:
                return False

        found = False
        for path_name, path in MSYS2_PATHS.items():
            if (
                Path(path).exists()
                and Path(path, "gcc.exe").exists()
                and Path(path, "mingw32-make.exe").exists()
            ):
                os.environ["PATH"] = f"{path};{os.environ['PATH']}"
                output_fn(f"Found GCC in {path}\n")
                max_retries = 3
                for i in range(max_retries):
                    time.sleep(1)  # Wait for PATH to update
                    if verify_gcc_make():
                        found = True
                        break
                    output_fn(f"Verification attempt {i+1} failed, retrying...\n")

        if found:
            output_fn("GCC/Make verified successfully\n")
            return

        output_fn("GCC/Make not found in system.\n")
        mingw_url = "https://github.com/skeeto/w64devkit/releases/download/v2.0.0/w64devkit-x64-2.0.0.exe"
        exe_path = project_root / "w64devkit.exe"

        class Spinner:
            def __init__(self):
                self.frames = itertools.cycle(["|", "/", "-", "\\"])
                self.last_time = 0
                self.min_interval = 0.1

            def next_frame(self):
                current_time = time.time()
                if current_time - self.last_time >= self.min_interval:
                    self.last_time = current_time
                    return next(self.frames)
                return None

        spinner = Spinner()

        try:
            if not exe_path.exists():
                output_fn(f"Downloading w64devkit... ")

                def progress_callback(count, block_size, total_size):
                    if frame := spinner.next_frame():
                        output_fn(
                            f"\rDownloading w64devkit... {frame} ({count*block_size*100//total_size}%)"
                        )

                urllib.request.urlretrieve(mingw_url, exe_path, progress_callback)
                output_fn("\rDownload complete!\n")

            output_fn("Extracting w64devkit... ")

            process = subprocess.Popen([str(exe_path)])

            while process.poll() is None:
                if frame := spinner.next_frame():
                    output_fn(
                        f"\rPlease click 'Extract' to extract w64devkit... {frame}"
                    )

            if process.returncode != 0:
                raise Exception(f"Extraction failed with code {process.returncode}")

            if exe_path.exists():
                exe_path.unlink()
            output_fn("\rExtraction complete!\n")

            mingw_path = str(project_root / "w64devkit/bin")
            os.environ["PATH"] = f"{mingw_path};{os.environ['PATH']}"
            output_fn("Verifying installation...\n")
            max_retries = 3
            for i in range(max_retries):
                time.sleep(1)
                if verify_gcc_make():
                    output_fn("w64devkit installed and verified successfully!\n")
                    return
                output_fn(f"Verification attempt {i+1} failed, retrying...\n")

            raise Exception(
                "Failed to verify w64devkit installation after multiple attempts"
            )

        except Exception as e:
            output_fn(f"Failed to install w64devkit: {str(e)}\n")
            output_fn("Make sure to have a stable internet connection, or\n")
            output_fn("Please install MSYS2 manually from https://www.msys2.org/\n")
            if exe_path.exists():
                exe_path.unlink()
            sys.exit(1)

    def _handle_process_output(self, process, output_fn):
        try:
            while True:
                line = process.stdout.readline() if process.stdout else ""
                if not line and process.poll() is not None:
                    break
                if line:
                    output_fn(line)

            if process.returncode != 0:
                output_fn(f"Process failed with exit code {process.returncode}\n")
            return process.returncode == 0
        except Exception as e:
            output_fn(f"Error monitoring process: {str(e)}\n")
            return False

    def _run_process(self, cmd, output_fn, cwd, on_complete=None, process_key=None):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True,
                cwd=cwd,
            )

            if process_key:
                self._active_processes[process_key] = process

            def monitor():
                success = self._handle_process_output(process, output_fn)
                if process_key and process_key in self._active_processes:
                    del self._active_processes[process_key]
                if not self.is_busy():
                    self._update_button_states(True)
                if on_complete:
                    on_complete(success)

            threading.Thread(target=monitor, daemon=True).start()

        except Exception as e:
            output_fn(f"Error starting process: {str(e)}\n")
            if on_complete:
                on_complete(False)

    def run_make(self, output_fn, target, cwd=project_root, on_complete=None):
        with self._lock:
            self._is_building = True
            self._build_states[target] = True
            self._update_button_states(False)

        def wrapped_complete(success):
            with self._lock:
                self._is_building = False
                if not self.is_busy():
                    self._update_button_states(True)
            if on_complete:
                on_complete(success)

        output_fn("\fBuilding...\n")
        self._run_process(f"{self.make_cmd} {target}", output_fn, cwd, wrapped_complete)

    def run_binary(self, output_fn, name, cwd=project_root):
        if (
            name in self._active_processes
            and self._active_processes[name].poll() is None
        ):
            return
        binary_name = f'{name}{".exe" if self.os_name == "windows" else ""}'
        if self.os_name != "windows":
            binary_path = f"bin/{binary_name}"
        else:
            binary_path = str(cwd / "bin" / binary_name)
        output_fn("\f")
        self._run_process(binary_path, output_fn, cwd, process_key=name)
        self._update_button_states(False)

    def build_and_run(self, output_fn, binary, target, cwd=project_root):
        if self.is_busy():
            return

        if not self._build_states.get(target, False):
            self.run_make(
                output_fn,
                target,
                cwd,
                on_complete=lambda success: success
                and self.run_binary(output_fn, binary, cwd),
            )
        else:
            self.run_binary(output_fn, binary, cwd)

    def _update_button_states(self, enabled):
        for callback in self._state_callbacks:
            callback(enabled)

    def is_busy(self):
        return self._is_building or bool(self._active_processes)

    def register_button_callback(self, callback):
        self._state_callbacks.add(callback)
        callback(not self.is_busy())

    def reset_build_status(self):
        with self._lock:
            self._build_states.update({k: False for k in self._build_states})

    def init(self, output_fn, cwd=project_root):
        threading.Thread(
            target=lambda: self._async_init(output_fn, cwd), daemon=True
        ).start()

    def _async_init(self, output_fn, cwd):
        self._is_building = True
        self._update_button_states(False)

        def chain_builds(success):
            if not success:
                self._is_building = False
                self._update_button_states(True)
                return

            if not Path(cwd, "results").exists():
                Path(cwd, "results").mkdir(parents=True, exist_ok=True)
            self.run_make(
                output_fn,
                "all",
                cwd,
                lambda s: (
                    output_fn(
                        "\fBuild complete! You can now press the Run button or modify the settings below.\n"
                    )
                    if s
                    else None
                ),
            )

        try:
            if self.os_name == "windows":
                self.verify_windows_environment(output_fn)
            chain_builds(True)
        except:
            self._is_building = False
            if not self.is_busy():
                self._update_button_states(True)

    def _cleanup_processes(self):
        for proc in self._active_processes.values():
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1)
                except:
                    proc.kill()


build_env = BuildEnvironment()

DISPLAY_BINARY_PROFILE = [
    ("Run", "main", "all"),
]
