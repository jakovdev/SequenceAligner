"""Creates enlarged dataset by multiplying the original data, used for testing purposes"""

from sys import stdout, platform
from argparse import ArgumentParser
from pathlib import Path
from contextlib import contextmanager
from time import time

if platform == "win32":
    import ctypes
    from ctypes import wintypes, windll, Structure, c_ulonglong, c_ulong, byref

    class MEMORYSTATUSEX(Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemLoad", wintypes.DWORD),
            ("ullTotalPhys", c_ulonglong),
            ("ullAvailPhys", c_ulonglong),
            ("ullTotalPageFile", c_ulonglong),
            ("ullAvailPageFile", c_ulonglong),
            ("ullTotalVirtual", c_ulonglong),
            ("ullAvailVirtual", c_ulonglong),
            ("ullAvailExtendedVirtual", c_ulonglong),
        ]


DEFAULT_MULTIPLIER = 16
UPDATE_INTERVAL_PERCENT = 5
MEMORY_SAFETY_FACTOR = 2.0  # Buffer for other processes


@contextmanager
def progress_bar():
    try:
        yield
    finally:
        stdout.write("\n")
        stdout.flush()


def get_available_memory():
    if platform == "win32":
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if not windll.kernel32.GlobalMemoryStatusEx(byref(stat)):
            raise OSError("Failed to get memory status")
        return stat.ullAvailPhys
    elif platform == "linux":
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemAvailable" in line:
                    return int(line.split()[1]) * 1024  # Convert KB to bytes
        raise OSError("Could not determine available memory")
    else:
        raise OSError(f"Unsupported platform: {platform}")


def check_memory_requirements(
    file_size, multiply_factor, reverse=True, skip_confirm=False
):
    available_memory = get_available_memory()
    data_multiplier = 2 if reverse else 1
    required_memory = (
        file_size * data_multiplier * multiply_factor * MEMORY_SAFETY_FACTOR
    )
    if not skip_confirm:
        print(f"Info: Available memory: {available_memory / (1024 ** 3):.1f}GB")
        print(f"Info: Required memory: {required_memory / (1024 ** 3):.1f}GB")

    if required_memory > available_memory:
        readable_required = required_memory / (1024**3)
        readable_available = available_memory / (1024**3)
        raise ValueError(
            f"Insufficient memory. Required: {readable_required:.1f}GB, "
            f"Available: {readable_available:.1f}GB. "
            f"Try reducing multiply_factor (current: {multiply_factor})"
        )


def enlarge_csv(
    multiply_factor=DEFAULT_MULTIPLIER,
    input_path=None,
    output_path=None,
    reverse=True,
    skip_header=True,
    skip_confirm=False,
) -> None:
    """
    Enlarge CSV file by multiplying its content.

    Args:
        multiply_factor: Number of times to multiply the data (default: 16)
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        reverse: If True, the reversed data will be appended to the original data
        skip_header: Set to true if CSV has a header that should be skipped
        skip_confirm: Skip confirmation prompt and remove info messages
    """

    if multiply_factor < 1:
        raise ValueError("Multiply factor must be positive")

    if input_path is None:
        raise ValueError("Input path was not sent properly (None)")

    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_path:
        raise ValueError("Output path was not sent properly (None)")

    file_size = input_file.stat().st_size
    check_memory_requirements(file_size, multiply_factor, reverse, skip_confirm)

    if not skip_confirm:
        input("Press Enter to continue or Ctrl+C to cancel...")
        begin_time = time()

    with open(input_file, "rb") as file:
        if skip_header:
            header = file.readline()
        else:
            header = b""
        data = file.read()

    data_size = len(data)
    total_size = len(header) + (data_size * 2 * multiply_factor)

    output_buffer = bytearray(total_size)
    current_pos = 0

    output_buffer[0 : len(header)] = header
    current_pos = len(header)

    if reverse:
        doubled_data = data + b"".join(data.splitlines(True)[::-1])
        doubled_size = len(doubled_data)
    else:
        doubled_data = data
        doubled_size = data_size

    # Pre-calculate update points (every 5%)
    update_interval = multiply_factor // (100 / UPDATE_INTERVAL_PERCENT)
    next_update = update_interval

    if not skip_confirm:
        print(f"Processing {multiply_factor} times the original dataset...")
        with progress_bar():
            for i in range(multiply_factor):
                if (i + 1) == next_update:
                    percentage = ((i + 1) * 100) // multiply_factor
                    stdout.write(f"\rMultiplying: {percentage}%/100%")
                    stdout.flush()
                    next_update += update_interval

                output_buffer[current_pos : current_pos + doubled_size] = doubled_data
                current_pos += doubled_size
        print("Writing output file...")
    else:
        for i in range(multiply_factor):
            output_buffer[current_pos : current_pos + doubled_size] = doubled_data
            current_pos += doubled_size

    with open(output_path, "wb") as file:
        file.write(output_buffer)

    if not skip_confirm:
        elapsed_time = time() - begin_time
        print("Done in {:.2f} seconds.".format(elapsed_time))


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Default paths
    input_path = project_root / "datasets" / "avpdb.csv"
    output_path = project_root / "datasets" / "avpdb_mt.csv"

    parser = ArgumentParser(
        description="Create enlarged dataset by multiplying the original data"
    )
    parser.add_argument(
        "multiplier",
        type=int,
        nargs="?",
        default=16,
        help="Number of times to multiply the data (default: 16)",
    )
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default=input_path,
        help="Path to input CSV file (default: avpdb.csv)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default=output_path,
        help="Path to output CSV file (default: avpdb_mega.csv)",
    )
    parser.add_argument(
        "--no-reverse",
        "-nr",
        action="store_false",
        dest="reverse",
        help="If set, the reversed data will not be appended to the original data",
    )
    parser.add_argument(
        "--no-header",
        "-nh",
        action="store_false",
        dest="skip_header",
        help="Set if CSV has no header to skip",
    )
    parser.add_argument(
        "--skip-confirm",
        "-sc",
        action="store_true",
        dest="skip_confirm",
        help="Skip confirmation prompt and remove info messages",
    )
    args = parser.parse_args()

    output_path_obj = Path(args.output_path)
    if output_path_obj.exists() and not args.skip_confirm:
        confirmation = input("Warning: Output file already exists. Overwrite? (y/N): ")
        if confirmation.lower() != "y":
            print("Operation aborted")
            exit(0)

    if not args.skip_confirm:
        print(f"Info: Using input path: {args.input_path}")
        print(f"Info: Using output path: {args.output_path}")
        print(f"Info: Using multiplier: {args.multiplier}")
        print(f"Info: Using reverse: {args.reverse}")
        print(f"Info: Skip header: {args.skip_header}")

    try:
        enlarge_csv(
            args.multiplier,
            args.input_path,
            args.output_path,
            args.reverse,
            args.skip_header,
            args.skip_confirm,
        )
    except (ValueError, FileNotFoundError, OSError) as e:
        print(f"Error: {e}")
        exit(1)
