#!/usr/bin/env python3
import argparse
import csv
import time
import multiprocessing as mp
import parasail
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Parasail sequence aligner")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", help="Output file (ignored)")
    parser.add_argument(
        "-T", "--threads", type=int, default=1, help="Number of processes"
    )
    parser.add_argument("-C", "--cpu", action="store_true", help="CPU mode (ignored)")
    parser.add_argument("-t", "--type", default="amino", help="Sequence type (ignored)")
    parser.add_argument("-D", "--flag-d", action="store_true", help="Flag D (ignored)")
    parser.add_argument("-B", "--flag-b", action="store_true", help="Flag B (ignored)")
    parser.add_argument("-m", "--matrix", default="blosum62", help="Scoring matrix")
    parser.add_argument(
        "-p",
        "--gap-penalty",
        type=int,
        default=4,
        help="Gap penalty for NW (both open and extend)",
    )
    parser.add_argument(
        "-s", "--gap-open", type=int, default=10, help="Gap open penalty"
    )
    parser.add_argument(
        "-e", "--gap-extend", type=int, default=1, help="Gap extend penalty"
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=["nw", "ga", "sw"],
        required=True,
        help="Alignment algorithm",
    )

    return parser.parse_args()


def load_sequences(input_file):
    sequences = []
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            if row and len(row) > 0:
                sequence = row[0].strip()
                if sequence:
                    sequences.append(sequence)
    return sequences


def get_parasail_matrix(matrix_name):
    matrix_name = matrix_name.lower()
    if matrix_name == "blosum62":
        return parasail.blosum62
    elif matrix_name == "blosum50":
        return parasail.blosum50
    elif matrix_name == "blosum80":
        return parasail.blosum80
    elif matrix_name == "pam250":
        return parasail.pam250
    else:
        # Default to blosum62 if unknown
        return parasail.blosum62


def align_pair(seq1, seq2, algorithm, matrix, gap_open, gap_extend):
    if algorithm == "nw" or algorithm == "ga":
        result = parasail.nw(seq1, seq2, gap_open, gap_extend, matrix)
    elif algorithm == "sw":
        result = parasail.sw(seq1, seq2, gap_open, gap_extend, matrix)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return result.score


def worker_process(args):
    (
        process_id,
        sequences,
        start_row,
        end_row,
        algorithm,
        matrix_name,
        gap_open,
        gap_extend,
    ) = args

    matrix = get_parasail_matrix(matrix_name)
    local_checksum = 0
    local_count = 0

    for i in range(start_row, min(end_row, len(sequences))):
        for j in range(i + 1, len(sequences)):
            score = align_pair(
                sequences[i], sequences[j], algorithm, matrix, gap_open, gap_extend
            )
            local_checksum += score
            local_count += 1

    return local_checksum, local_count


def align_sequences(
    sequences, algorithm, matrix_name, gap_open, gap_extend, num_processes
):
    n = len(sequences)
    total_alignments = n * (n - 1) // 2

    print(f"Found {n} sequences")
    print(f"Will perform {total_alignments} pairwise alignments")
    print(f"Average sequence length: {sum(len(seq) for seq in sequences) / n:.1f}")
    print(f"Threads: {num_processes}")
    print("CUDA: Disabled")

    start_time = time.time()

    if num_processes == 1:
        matrix = get_parasail_matrix(matrix_name)
        local_checksum = 0
        alignment_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                score = align_pair(
                    sequences[i], sequences[j], algorithm, matrix, gap_open, gap_extend
                )
                local_checksum += score
                alignment_count += 1

                # Print progress every 1% or every 10000 alignments
                # if (
                #     alignment_count % max(1, min(10000, total_alignments // 100)) == 0
                #     or alignment_count == total_alignments
                # ):
                #     progress = (alignment_count / total_alignments) * 100
                #     print(f"\rAligning sequences: {progress:.0f}%", end="", flush=True)

        print()
        total_checksum = local_checksum * 2
    else:
        rows_per_process = n // num_processes
        process_args = []

        for p in range(num_processes):
            start_row = p * rows_per_process
            if p == num_processes - 1:
                end_row = n
            else:
                end_row = (p + 1) * rows_per_process

            if start_row < n:
                process_args.append(
                    (
                        p,
                        sequences,
                        start_row,
                        end_row,
                        algorithm,
                        matrix_name,
                        gap_open,
                        gap_extend,
                    )
                )

        # print("\rAligning sequences: 0%", end="", flush=True)

        with mp.Pool(processes=len(process_args)) as pool:
            results = pool.map(worker_process, process_args)

        # print("\rAligning sequences: 100%", end="", flush=True)
        print()

        total_checksum = sum(checksum for checksum, _ in results) * 2

    end_time = time.time()
    compute_time = end_time - start_time

    return None, total_checksum, compute_time


def main():
    args = parse_args()

    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass

    if args.algorithm == "nw":
        gap_open = args.gap_penalty
        gap_extend = args.gap_penalty
    else:  # ga, sw
        gap_open = args.gap_open
        gap_extend = args.gap_extend

    try:
        sequences = load_sequences(args.input)
    except Exception as e:
        print(f"Error loading sequences: {e}", file=sys.stderr)
        sys.exit(1)

    if not sequences:
        print("No sequences found in input file", file=sys.stderr)
        sys.exit(1)

    try:
        _, checksum, compute_time = align_sequences(
            sequences, args.algorithm, args.matrix, gap_open, gap_extend, args.threads
        )

        total_alignments = len(sequences) * (len(sequences) - 1) // 2
        alignments_per_sec = total_alignments / compute_time

        print("\nPerformance Summary")
        print("Timing breakdown:")
        print(f"Compute: {compute_time:.3f} sec (100.0%)")
        print(f"I/O: 0.000 sec (0.0%)")
        print(f"Total: {compute_time:.3f} sec")
        print(f"Alignments per second: {alignments_per_sec:.2f}")

        if args.threads > 1:
            avg_time_per_thread = compute_time / args.threads
            aps_per_thread = alignments_per_sec / args.threads
            print(f"Average time per thread: {avg_time_per_thread:.3f} sec")
            print(f"Alignments per second per thread: {aps_per_thread:.2f}")

        print(f"Matrix checksum: {checksum}")

    except Exception as e:
        print(f"Error during alignment: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
