#!/usr/bin/env python3
import h5py
import numpy as np
import argparse


def compare_matrices(file1, file2):
    print(f"Comparing matrices in {file1} and {file2}...")

    with h5py.File(file1, "r") as f1, h5py.File(file2, "r") as f2:
        matrix1 = f1["/similarity_matrix"]
        matrix2 = f2["/similarity_matrix"]

        if matrix1.shape != matrix2.shape:
            print(f"Matrix shapes differ: {matrix1.shape} vs {matrix2.shape}")
            return

        shape = matrix1.shape
        differences_found = False

        chunk_size = 1024

        for i in range(0, shape[0], chunk_size):
            end_i = min(i + chunk_size, shape[0])
            for j in range(0, shape[1], chunk_size):
                end_j = min(j + chunk_size, shape[1])

                chunk1 = matrix1[i:end_i, j:end_j]
                chunk2 = matrix2[i:end_i, j:end_j]

                diff_indices = np.where(chunk1 != chunk2)
                for idx in range(len(diff_indices[0])):
                    row = diff_indices[0][idx] + i
                    col = diff_indices[1][idx] + j
                    differences_found = True
                    print(f"Difference at position ({row}, {col}):")
                    print(
                        f"  {file1}: {chunk1[diff_indices[0][idx], diff_indices[1][idx]]}"
                    )
                    print(
                        f"  {file2}: {chunk2[diff_indices[0][idx], diff_indices[1][idx]]}"
                    )

        if not differences_found:
            print("Matrices are identical.")

        if "checksum" in matrix1.attrs and "checksum" in matrix2.attrs:
            checksum1 = matrix1.attrs["checksum"]
            checksum2 = matrix2.attrs["checksum"]
            print(f"Checksums: {file1}={checksum1}, {file2}={checksum2}")
            if checksum1 != checksum2:
                print("Checksums differ!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare similarity matrices in two HDF5 files."
    )
    parser.add_argument("file1", help="Path to first HDF5 file")
    parser.add_argument("file2", help="Path to second HDF5 file")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        compare_matrices(args.file1, args.file2)
    except Exception as e:
        print(f"Error: {str(e)}")
