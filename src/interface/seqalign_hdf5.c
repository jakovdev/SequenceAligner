#include "interface/seqalign_hdf5.h"

#include <args.h>
#include <hdf5.h>
#include <print.h>
#include <string.h>

#include "bio/sequence/sequences.h"
#include "bio/types.h"
#include "interface/seqalign_cuda.h"
#include "io/mmap.h"
#include "system/os.h"
#include "system/memory.h"
#include "util/benchmark.h"

static struct {
	const char *path;
	hid_t file_id;
	hid_t matrix_id;
	hid_t sequences_id;
	s32 *matrix;
	size_t matrix_b;
	s64 checksum;
	struct MMapMatrix mmap;
	s32 chunk_dim;
	u8 compression;
	bool disabled;
	bool mode_mmap;
	bool triangular;
	bool is_init;
} g_h5;

#define H5_MIN_CHUNK_SIZE (1 << 8)
#define H5_MAX_CHUNK_SIZE (1 << 12)
static void h5_chunk_dimensions_calculate(void)
{
	if (SEQS_N <= H5_MIN_CHUNK_SIZE) {
		g_h5.chunk_dim = SEQS_N;
		return;
	}

	s32 chunk_dim = 64;
	size_t square = (size_t)(chunk_dim * chunk_dim) * sizeof(chunk_dim);
	size_t target_bytes = (2 * MiB) / (1 + (size_t)g_h5.compression / 3);
	while (chunk_dim < SEQS_N && square < target_bytes)
		chunk_dim *= 2;
	if (chunk_dim > SEQS_N || square > target_bytes)
		chunk_dim /= 2;

	chunk_dim = max(chunk_dim, H5_MIN_CHUNK_SIZE);
	chunk_dim = min(chunk_dim, H5_MAX_CHUNK_SIZE);
	chunk_dim = min(chunk_dim, SEQS_N);
	g_h5.chunk_dim = chunk_dim;
}

static void h5_file_close(void);

bool h5_open(void)
{
	if unlikely (g_h5.is_init) {
		pdev("Call h5_close() before calling h5_open() again");
		perr("Internal error initializing HDF5 storage");
		pabort();
	}

	g_h5.file_id = H5I_INVALID_HID;
	g_h5.matrix_id = H5I_INVALID_HID;
	g_h5.sequences_id = H5I_INVALID_HID;

	if (g_h5.disabled) {
		g_h5.is_init = true;
		return true;
	}

	if unlikely (SEQS_N < SEQ_N_MIN || !SEQS) {
		pdev("Sequences not initialized before h5_open()");
		perr("Internal error initializing HDF5 storage");
		pabort();
	}

	const size_t dim_size = (size_t)SEQS_N;
	h5_chunk_dimensions_calculate();
	bench_io_start();

	const size_t safe = available_memory() * 3 / 4;
	size_t bytes = bytesof(g_h5.matrix, dim_size * dim_size);
	bool device_limited = arg_mode_cuda() && !cuda_memory(bytes);
	if (device_limited || bytes > safe) {
		bytes = bytesof(g_h5.matrix, dim_size * (dim_size - 1) / 2);
		g_h5.mode_mmap = bytes > safe;
		g_h5.triangular = true;
		pverb("Using triangular matrix storage");
	}

	if (g_h5.mode_mmap) {
		pinfo("Matrix size exceeds memory limits");
		if (!mmap_matrix_open(&g_h5.mmap, dim_size))
			return false;
		g_h5.matrix = g_h5.mmap.matrix;
		g_h5.matrix_b = g_h5.mmap.bytes;
	} else {
		MALLOC_AL(g_h5.matrix, PAGE_SIZE, bytes);
		if unlikely (!g_h5.matrix) {
			perr("Out of memory allocating similarity matrix");
			return false;
		}
		memset(g_h5.matrix, 0, bytes);
		g_h5.matrix_b = bytes;
	}
	pverb("HDF5 matrix size: %zu x %zu", dim_size, dim_size);

	hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
	H5Pset_alignment(fapl, 4096, 4096);
	g_h5.file_id = H5Fcreate(g_h5.path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
	H5Pclose(fapl);
	if unlikely (g_h5.file_id < 0) {
		perr("Failed to create HDF5 file: %s", file_name(g_h5.path));
		h5_file_close();
		return false;
	}

	hsize_t matrix_dims[2] = { dim_size, dim_size };
	hid_t matrix_space = H5Screate_simple(2, matrix_dims, NULL);
	if unlikely (matrix_space < 0) {
		perr("Failed to create HDF5 dataspace for similarity matrix");
		h5_file_close();
		return false;
	}

	hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
	if (dim_size > H5_MIN_CHUNK_SIZE) {
		hsize_t chunk_dims[2] = { (hsize_t)g_h5.chunk_dim,
					  (hsize_t)g_h5.chunk_dim };
		H5Pset_chunk(plist_id, 2, chunk_dims);
		pverbl("HDF5 chunk size: " Ps32 " x " Ps32, g_h5.chunk_dim,
		       g_h5.chunk_dim);

		if (g_h5.compression > 0)
			H5Pset_deflate(plist_id, g_h5.compression);
	}
	g_h5.matrix_id = H5Dcreate2(g_h5.file_id, "/similarity_matrix",
				    H5T_STD_I32LE, matrix_space, H5P_DEFAULT,
				    plist_id, H5P_DEFAULT);
	H5Sclose(matrix_space);
	H5Pclose(plist_id);
	if unlikely (g_h5.matrix_id < 0) {
		perr("Failed to create HDF5 dataset for similarity matrix");
		h5_file_close();
		return false;
	}

	pverb("Storing %zu sequences in HDF5 file", dim_size);

	hsize_t seq_dims[1] = { dim_size };
	hid_t seq_space = H5Screate_simple(1, seq_dims, NULL);
	if unlikely (seq_space < 0) {
		perr("Failed to create HDF5 dataspace for sequences");
		h5_file_close();
		return false;
	}

	hid_t string_type = H5Tcopy(H5T_C_S1);
	H5Tset_size(string_type, H5T_VARIABLE);
	g_h5.sequences_id = H5Dcreate2(g_h5.file_id, "/sequences", string_type,
				       seq_space, H5P_DEFAULT, H5P_DEFAULT,
				       H5P_DEFAULT);
	if unlikely (g_h5.sequences_id < 0) {
		perr("Failed to create HDF5 dataset for sequences");
		H5Sclose(seq_space);
		H5Tclose(string_type);
		h5_file_close();
		return false;
	}

	const char **MALLOCA(seq_data, dim_size);
	if unlikely (!seq_data) {
		perr("Out of memory allocating HDF5 sequence data");
		H5Sclose(seq_space);
		H5Tclose(string_type);
		h5_file_close();
		return false;
	}

	for (s32 i = 0; i < SEQS_N; i++)
		seq_data[i] = SEQS[i].letters;

	herr_t status = H5Dwrite(g_h5.sequences_id, string_type, H5S_ALL,
				 H5S_ALL, H5P_DEFAULT, seq_data);
	free(seq_data);
	if unlikely (status < 0) {
		perr("Failed to write sequence data to HDF5 dataset");
		H5Sclose(seq_space);
		H5Tclose(string_type);
		h5_file_close();
		return false;
	}

	H5Sclose(seq_space);
	H5Tclose(string_type);
	bench_io_end();
	g_h5.is_init = true;
	return true;
}

void h5_matrix_column_set(s32 col, const s32 PRS(values, SEQS_N))
{
	if (g_h5.disabled)
		return;

	if (!g_h5.matrix || col < 0 || col >= SEQS_N)
		unreachable_release();

	if (g_h5.triangular) {
		memcpy(g_h5.matrix + ((s64)col * (col - 1)) / 2, values,
		       bytesof(g_h5.matrix, (size_t)col));
	} else {
		const s64 dim = SEQS_N;
		for (s32 row = 0; row < col; row++) {
			g_h5.matrix[dim * row + col] = values[row];
			g_h5.matrix[dim * col + row] = values[row];
		}
	}
}

void h5_checksum_set(s64 checksum)
{
	if unlikely (!g_h5.is_init) {
		pdev("HDF5 file not opened when setting checksum");
		perr("Internal error setting HDF5 checksum");
		pabort();
	}

	g_h5.checksum = checksum;
}

s64 h5_checksum(void)
{
	if unlikely (!g_h5.is_init) {
		pdev("HDF5 file not opened when getting checksum");
		perr("Internal error getting HDF5 checksum");
		pabort();
	}

	return g_h5.checksum;
}

static void h5_store_checksum(void);
static void h5_flush_matrix(void);

void h5_close(int skip_flush)
{
	if unlikely (!g_h5.is_init) {
		pdev("HDF5 file not opened or already closed");
		perr("Internal error closing HDF5 file");
		pabort();
	}

	if likely (!skip_flush) {
		psection("Finalizing Results");
		pinfo("Matrix checksum: " Ps64, g_h5.checksum);
	}

	if (!g_h5.disabled) {
		bench_io_start();
		if likely (!skip_flush) {
			pinfol("Writing results to HDF5");
			h5_store_checksum();
			h5_flush_matrix();
		}

		h5_file_close();
		bench_io_end();
	}

	if likely (!skip_flush)
		bench_io_print();
	g_h5.is_init = false;
}

s32 *h5_matrix_data(void)
{
	if unlikely (!g_h5.is_init) {
		pdev("HDF5 file not opened when getting matrix data");
		perr("Internal error getting HDF5 matrix data");
		pabort();
	}

	return g_h5.matrix;
}

size_t h5_matrix_bytes(void)
{
	if unlikely (!g_h5.is_init) {
		pdev("HDF5 file not opened when getting matrix size");
		perr("Internal error getting HDF5 matrix size");
		pabort();
	}

	return g_h5.matrix_b;
}

static void h5_file_close(void)
{
	if (g_h5.mode_mmap) {
		mmap_matrix_close(&g_h5.mmap);
	} else {
		if (g_h5.matrix)
			free_aligned(g_h5.matrix);
	}
	g_h5.matrix = NULL;
	g_h5.matrix_b = 0;

	if (g_h5.sequences_id > 0) {
		H5Dclose(g_h5.sequences_id);
		g_h5.sequences_id = H5I_INVALID_HID;
	}

	if (g_h5.matrix_id > 0) {
		H5Dclose(g_h5.matrix_id);
		g_h5.matrix_id = H5I_INVALID_HID;
	}

	if (g_h5.file_id > 0) {
		H5Fclose(g_h5.file_id);
		g_h5.file_id = H5I_INVALID_HID;
	}
}

static void h5_store_checksum(void)
{
	hid_t attr_space = H5Screate(H5S_SCALAR);
	if unlikely (attr_space < 0) {
		perr("Failed to create HDF5 dataspace for checksum attribute");
		return;
	}

	hid_t attr_id = H5Acreate2(g_h5.matrix_id, "checksum", H5T_STD_I64LE,
				   attr_space, H5P_DEFAULT, H5P_DEFAULT);
	if unlikely (attr_id < 0) {
		perr("Failed to create checksum attribute");
		H5Sclose(attr_space);
		return;
	}

	herr_t status = H5Awrite(attr_id, H5T_NATIVE_INT64, &g_h5.checksum);
	H5Aclose(attr_id);
	H5Sclose(attr_space);
	if unlikely (status < 0)
		perr("Failed to write checksum attribute");
}

static void h5_flush_matrix(void)
{
	if (!g_h5.triangular) {
		herr_t status = H5Dwrite(g_h5.matrix_id, H5T_NATIVE_INT32,
					 H5S_ALL, H5S_ALL, H5P_DEFAULT,
					 g_h5.matrix);
		if unlikely (status < 0)
			perr("Failed to write similarity matrix to HDF5");
		return;
	}

	pinfo("Converting triangular matrix to HDF5 format");

	hid_t file_space = H5Dget_space(g_h5.matrix_id);
	if unlikely (file_space < 0) {
		perr("Failed to get HDF5 dataspace for similarity matrix");
		return;
	}

	const size_t available_mem = available_memory();
	if unlikely (!available_mem) {
		perr("Failed to retrieve available memory");
		H5Sclose(file_space);
		return;
	}

	const s64 dim = SEQS_N;
	const size_t row_bytes = bytesof(g_h5.matrix, (size_t)dim);
	const s32 max_rows = (s32)(available_mem / (4 * row_bytes));
	s32 chunk_size = g_h5.chunk_dim > 4 ? g_h5.chunk_dim : 4;
	if (chunk_size > max_rows && max_rows > 4)
		chunk_size = max_rows;

	s32 *MALLOC_AL(buf, PAGE_SIZE, row_bytes * (size_t)chunk_size);
	if unlikely (!buf) {
		pwarn("Out of memory, trying minimal amount for conversion");
		chunk_size = 1;
		MALLOC_AL(buf, CACHE_LINE, row_bytes * (size_t)chunk_size);
		if unlikely (!buf) {
			perr("Out of memory, aborting conversion");
			H5Sclose(file_space);
			return;
		}
	}
	memset(buf, 0, row_bytes * (size_t)chunk_size);

	ppercent(0, "Converting to HDF5");
	for (s32 off = 0; off < SEQS_N; off += chunk_size) {
		s32 end = min(off + chunk_size, SEQS_N);

		for (s32 i = off; i < end; i++) {
			s64 row = dim * (i - off);

			for (s32 j = i + 1; j < SEQS_N; j++)
				buf[row + j] = g_h5.matrix[matrix_index(i, j)];

			for (s32 j = 0; j < i; j++) {
				if (j >= off)
					buf[row + j] = buf[dim * (j - off) + i];
				else
					buf[row + j] =
						g_h5.matrix[matrix_index(j, i)];
			}
		}

		s32 rows = end - off;
		hsize_t start[2] = { (hsize_t)off, 0 };
		hsize_t count[2] = { (hsize_t)rows, (size_t)dim };
		H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL,
				    count, NULL);

		hsize_t mem_dims[2] = { (hsize_t)rows, (size_t)dim };
		hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);
		if unlikely (mem_space < 0) {
			perr("Failed to create memory dataspace for matrix chunk");
			H5Sclose(file_space);
			free_aligned(buf);
			return;
		}

		herr_t status = H5Dwrite(g_h5.matrix_id, H5T_NATIVE_INT32,
					 mem_space, file_space, H5P_DEFAULT,
					 buf);
		H5Sclose(mem_space);
		if unlikely (status < 0) {
			perr("Failed to write chunk to HDF5");
			H5Sclose(file_space);
			free_aligned(buf);
			return;
		}

		pproport(end / SEQS_N, "Converting to HDF5");
	}

	ppercent(100, "Converting to HDF5");
	H5Sclose(file_space);
	free_aligned(buf);
}

ARG_EXTERN(disable_cuda);

ARGUMENT(disable_write) = {
	.opt = 'W',
	.lopt = "no-write",
	.help = "Disable writing to output file",
	.set = &g_h5.disabled,
	.help_order = ARG_ORDER_AFTER(ARG(disable_cuda)),
};

static void print_output_path(void)
{
	if (g_h5.disabled)
		pwarnm("Output: Ignored");
	else
		pinfom("Output: %s", file_name(g_h5.path));
}

static struct arg_callback validate_output_path(void)
{
	if (g_h5.disabled)
		return ARG_VALID();

	if (path_file_exists(g_h5.path)) {
		pwarn("Output file already exists: %s", file_name(g_h5.path));
		if (!print_yN("Do you want to DELETE it?"))
			return ARG_INVALID(
				"Output file exists and will not be overwritten");
		if (remove(g_h5.path) != 0)
			return ARG_INVALID(
				"Failed to delete existing output file");
		pinfo("Deleted existing output file");
	}

	if (!path_directories_create(g_h5.path))
		return ARG_INVALID(
			"Failed to create directories for output file");

	return ARG_VALID();
}

ARG_EXTERN(input_path);

ARGUMENT(output_path) = {
	.opt = 'o',
	.lopt = "output",
	.help = "Output file path: HDF5 format",
	.param = "FILE",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = &g_h5.path,
	.parse_callback = parse_path,
	.validate_callback = validate_output_path,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_AFTER(ARG(input_path)),
	.action_callback = print_output_path,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(input_path)),
	.help_order = ARG_ORDER_AFTER(ARG(input_path)),
	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(disable_write)),
};

ARG_PARSE_L(compression, 10, u8, (u8), (val < 0 || val > 9),
	    "Compression level must be between 0-9")

static void print_compression(void)
{
	pinfom("Compression: " Pu8, g_h5.compression);
}

ARG_EXTERN(filter_threshold);

ARGUMENT(compression) = {
	.opt = 'z',
	.lopt = "compression",
	.help = "Compression level for HDF5 datasets [0-9]",
	.param = "N",
	.param_req = ARG_PARAM_REQUIRED,
	.dest = &g_h5.compression,
	.parse_callback = parse_compression,
	.action_callback = print_compression,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(filter_threshold)),
	.help_order = ARG_ORDER_AFTER(ARG(filter_threshold)),
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(output_path)),
};
