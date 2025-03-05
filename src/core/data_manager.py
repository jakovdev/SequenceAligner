import numpy as np
import polars as pl
from pathlib import Path
import h5py
import tempfile
import os
from typing import List, Tuple, Dict, Optional, Iterator, Union
import shutil
import logging

from .sequence import Sequence, SequenceCollection
from .config import Config

logger = logging.getLogger(__name__)


class DataManager:

    def __init__(self, config: Config):
        self.config = config
        self.temp_dir = None
        self.hdf5_path = None
        self._setup_temp_storage()

    def _setup_temp_storage(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="seqaligner_")
        self.hdf5_path = os.path.join(self.temp_dir, "sequence_data.h5")
        logger.info(f"Temporary data directory created at {self.temp_dir}")

    def __del__(self) -> None:
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Removed temporary directory {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory: {e}")

    def load_sequences(self, file_path: str) -> SequenceCollection:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        format_type = self._detect_file_format(file_path)
        if format_type == "fasta":
            return self._load_fasta_sequences(file_path)
        elif format_type in ["csv", "tsv"]:
            return self._load_tabular_sequences(file_path, format_type)
        else:
            raise ValueError(f"Unsupported file format: {format_type}")

    def _detect_file_format(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".fa", ".fasta"]:
            return "fasta"
        elif ext == ".csv":
            return "csv"
        elif ext == ".tsv":
            return "tsv"

        # If extension is ambiguous, try to peek at content
        try:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
                if first_line.startswith(">"):
                    return "fasta"
                elif "," in first_line:
                    return "csv"
                elif "\t" in first_line:
                    return "tsv"
        except Exception as e:
            logger.warning(f"Error detecting file format: {e}")

        # Default to CSV
        return "csv"

    def _load_fasta_sequences(self, file_path: str) -> SequenceCollection:
        from Bio import SeqIO

        if os.path.getsize(file_path) > 10_000_000:
            logger.info(f"Loading large FASTA file: {file_path}")
            return self._load_large_fasta(file_path)

        sequences = SequenceCollection()
        for record in SeqIO.parse(file_path, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)
            min_len = self.config.analysis.minimum_length
            max_len = self.config.analysis.maximum_length

            if len(sequence) < min_len or len(sequence) > max_len:
                continue

            sequences.add_sequence(Sequence(sequence=sequence, seq_id=seq_id))

        return sequences

    def _load_large_fasta(self, file_path: str) -> SequenceCollection:
        from Bio import SeqIO

        h5_path = os.path.join(self.temp_dir, "sequences.h5")

        with h5py.File(h5_path, "w") as h5file:
            sequences_ds = h5file.create_dataset(
                "sequences",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.special_dtype(vlen=str),
            )

            ids_ds = h5file.create_dataset(
                "ids", shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str)
            )

            chunk_size = 1000
            seq_buffer = []
            id_buffer = []
            total_seqs = 0

            for record in SeqIO.parse(file_path, "fasta"):
                seq_id = record.id
                sequence = str(record.seq)

                min_len = self.config.analysis.minimum_length
                max_len = self.config.analysis.maximum_length

                if len(sequence) < min_len or len(sequence) > max_len:
                    continue

                seq_buffer.append(sequence)
                id_buffer.append(seq_id)

                if len(seq_buffer) >= chunk_size:
                    new_size = total_seqs + len(seq_buffer)
                    sequences_ds.resize((new_size,))
                    ids_ds.resize((new_size,))

                    sequences_ds[total_seqs:new_size] = seq_buffer
                    ids_ds[total_seqs:new_size] = id_buffer

                    total_seqs = new_size
                    seq_buffer = []
                    id_buffer = []

            if seq_buffer:
                new_size = total_seqs + len(seq_buffer)
                sequences_ds.resize((new_size,))
                ids_ds.resize((new_size,))

                sequences_ds[total_seqs:new_size] = seq_buffer
                ids_ds[total_seqs:new_size] = id_buffer

        return HDF5SequenceCollection(h5_path)

    def _load_tabular_sequences(
        self, file_path: str, format_type: str
    ) -> SequenceCollection:
        delimiter = "," if format_type == "csv" else "\t"

        file_size = os.path.getsize(file_path)
        memory_limit = self.config.performance.memory_bytes

        if file_size < memory_limit / 2:  # safety margin
            logger.info(f"Loading file directly into memory: {file_path}")
            return self._load_small_tabular(file_path, delimiter)
        else:
            logger.info(f"File size exceeds memory limit. Using streaming: {file_path}")
            return self._load_large_tabular(file_path, delimiter)

    def _load_small_tabular(self, file_path: str, delimiter: str) -> SequenceCollection:
        df = pl.read_csv(file_path, separator=delimiter)
        required_cols = ["sequence"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"File must contain column: {required_cols}")

        collection = SequenceCollection()

        seq_col = df.columns.index("sequence")
        label_col = df.columns.index("label") if "label" in df.columns else None
        id_col = df.columns.index("id") if "id" in df.columns else None
        data = df.to_numpy()

        for row in data:
            sequence = row[seq_col]
            label = int(row[label_col]) if label_col is not None else None
            seq_id = row[id_col] if id_col is not None else None

            try:
                seq_obj = Sequence(sequence, seq_id, label)
                collection.add_sequence(seq_obj)
            except ValueError as e:
                logger.warning(f"Skipping invalid sequence: {e}")

        return collection

    def _load_large_tabular(self, file_path: str, delimiter: str) -> SequenceCollection:
        with h5py.File(self.hdf5_path, "w") as hdf:
            # Create dataset placeholders
            hdf.create_group("sequences")
            hdf.create_dataset("labels", (0,), maxshape=(None,), dtype="i4")
            hdf.create_dataset("ids", (0,), maxshape=(None,), dtype=h5py.string_dtype())

            batch_size = 10000
            reader = pl.read_csv_batched(
                file_path, batch_size=batch_size, separator=delimiter
            )

            seq_count = 0
            for batch_idx, batch_df in enumerate(reader):
                logger.info(f"Processing batch {batch_idx+1}")
                sequences = batch_df["sequence"].to_list()
                labels = (
                    batch_df["label"].to_list()
                    if "label" in batch_df.columns
                    else [None] * len(sequences)
                )
                ids = (
                    batch_df["id"].to_list()
                    if "id" in batch_df.columns
                    else [f"seq_{i+seq_count}" for i in range(len(sequences))]
                )
                valid_seqs = []
                valid_labels = []
                valid_ids = []

                for seq, label, seq_id in zip(sequences, labels, ids):
                    try:
                        if all(aa in Sequence.VALID_RESIDUES for aa in seq):
                            valid_seqs.append(seq)
                            valid_labels.append(
                                label if label is not None else -1
                            )  # -1 for missing label
                            valid_ids.append(
                                seq_id
                                if seq_id is not None
                                else f"seq_{seq_count+len(valid_seqs)}"
                            )
                    except Exception as e:
                        logger.warning(f"Skipping invalid sequence: {e}")

                if valid_seqs:
                    current_size = hdf["labels"].shape[0]
                    new_size = current_size + len(valid_seqs)

                    hdf["labels"].resize((new_size,))
                    hdf["ids"].resize((new_size,))
                    hdf["labels"][current_size:new_size] = valid_labels
                    hdf["ids"][current_size:new_size] = valid_ids

                    for i, seq in enumerate(valid_seqs):
                        seq_id = valid_ids[i]
                        seq_dataset = hdf["sequences"].create_dataset(
                            str(current_size + i), data=np.array(list(seq), dtype="S1")
                        )
                        seq_dataset.attrs["id"] = seq_id
                        seq_dataset.attrs["label"] = valid_labels[i]

                    seq_count += len(valid_seqs)

            logger.info(f"Loaded {seq_count} valid sequences into HDF5 storage")

        return HDF5SequenceCollection(self.hdf5_path)

    def get_temp_file_path(self, filename: str) -> str:
        return os.path.join(self.temp_dir, filename)


class HDF5SequenceCollection(SequenceCollection):

    def __init__(self, hdf5_path: str):
        super().__init__()
        self.hdf5_path = hdf5_path
        self._length = self._get_length()

    def _get_length(self) -> int:
        with h5py.File(self.hdf5_path, "r") as hdf:
            return len(hdf["labels"])

    def __len__(self) -> int:
        return self._length

    def get_sequences(self) -> List[Sequence]:
        sequences = []
        with h5py.File(self.hdf5_path, "r") as hdf:
            for i in range(self._length):
                seq_dataset = hdf["sequences"][str(i)]
                sequence = "".join(s.decode("utf-8") for s in seq_dataset[()])
                seq_id = seq_dataset.attrs["id"]
                label = (
                    int(seq_dataset.attrs["label"])
                    if seq_dataset.attrs["label"] >= 0
                    else None
                )
                sequences.append(Sequence(sequence, seq_id, label))
        return sequences

    def get_raw_sequences(self) -> List[str]:
        sequences = []
        with h5py.File(self.hdf5_path, "r") as hdf:
            for i in range(self._length):
                seq_data = hdf["sequences"][str(i)][()]
                sequence = "".join(s.decode("utf-8") for s in seq_data)
                sequences.append(sequence)
        return sequences

    def get_sequence_batch(self, start_idx: int, batch_size: int) -> List[Sequence]:
        end_idx = min(start_idx + batch_size, self._length)
        sequences = []

        with h5py.File(self.hdf5_path, "r") as hdf:
            for i in range(start_idx, end_idx):
                seq_dataset = hdf["sequences"][str(i)]
                sequence = "".join(s.decode("utf-8") for s in seq_dataset[()])
                seq_id = seq_dataset.attrs["id"]
                label = (
                    int(seq_dataset.attrs["label"])
                    if seq_dataset.attrs["label"] >= 0
                    else None
                )
                sequences.append(Sequence(sequence, seq_id, label))

        return sequences

    def get_labels(self) -> np.ndarray:
        with h5py.File(self.hdf5_path, "r") as hdf:
            labels = np.array(hdf["labels"])
            # Convert -1 (missing) to None
            mask = labels == -1
            result = labels.copy()
            if np.any(mask):
                return np.ma.array(result, mask=mask)
            return result

    def get_ids(self) -> List[str]:
        with h5py.File(self.hdf5_path, "r") as hdf:
            return [id.decode("utf-8") for id in hdf["ids"]]

    def iterator(self, batch_size: int = 100) -> Iterator[List[Sequence]]:
        for start in range(0, self._length, batch_size):
            yield self.get_sequence_batch(start, batch_size)

    def add_sequence(self, sequence: Sequence) -> None:
        with h5py.File(self.hdf5_path, "a") as hdf:
            current_size = len(hdf["labels"])

            hdf["labels"].resize((current_size + 1,))
            hdf["ids"].resize((current_size + 1,))
            hdf["labels"][current_size] = (
                sequence.label if sequence.label is not None else -1
            )
            hdf["ids"][current_size] = (
                sequence.id if sequence.id is not None else f"seq_{current_size}"
            )

            seq_dataset = hdf["sequences"].create_dataset(
                str(current_size), data=np.array(list(sequence.sequence), dtype="S1")
            )
            seq_dataset.attrs["id"] = (
                sequence.id if sequence.id is not None else f"seq_{current_size}"
            )
            seq_dataset.attrs["label"] = (
                sequence.label if sequence.label is not None else -1
            )

            self._length += 1

    def add_sequences(self, sequences: List[Sequence]) -> None:
        for seq in sequences:
            self.add_sequence(seq)
