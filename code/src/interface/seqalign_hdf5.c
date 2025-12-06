#include "interface/seqalign_hdf5.h"

#include <errno.h>
#include <hdf5.h>
#include <string.h>

#include "bio/types.h"
#include "io/files.h"
#include "interface/seqalign_cuda.h"
#include "system/os.h"
#include "system/memory.h"
#include "util/args.h"
#include "util/benchmark.h"
#include "util/print.h"

#ifdef USE_CUDA
#include "host_interface.h"
#endif

static struct {
	hid_t file_id;
	hid_t matrix_id;
	hid_t sequences_id;
	hid_t lengths_id;
	s32 *matrix;
	size_t matrix_b;
	struct FileScoreMatrix mmap;
	s64 mat_dim;
	s64 checksum;
	u8 compression;
	bool mode_write;
	bool mode_mmap;
	bool triangular;
	bool is_init;
} g_h5 = { 0 };

static void h5_file_close(void);
static s32 h5_chunk_dimensions_calculate(void);

#define H5_SEQUENCE_BATCH_SIZE (1 << 12)
#define H5_MIN_CHUNK_SIZE (1 << 7)
#define H5_MAX_CHUNK_SIZE (H5_MIN_CHUNK_SIZE << 7)
#define ALIGN_POW2(value, pow2) \
	(((value) + ((pow2 >> 1) - 1)) / (pow2)) * (pow2)

bool h5_open(const char *file_path, sequence_t *seqs, s32 seq_n)
{
	g_h5.file_id = H5I_INVALID_HID;
	g_h5.matrix_id = H5I_INVALID_HID;
	g_h5.sequences_id = H5I_INVALID_HID;
	g_h5.lengths_id = H5I_INVALID_HID;
	g_h5.mode_write = arg_mode_write();
	g_h5.mat_dim = seq_n;

	if (!g_h5.mode_write) {
		g_h5.is_init = true;
		return true;
	}

	perr_context("HDF5");

	if (g_h5.mat_dim < SEQUENCE_COUNT_MIN) {
		perr("Matrix size is too small");
		return false;
	}

	if (!file_path || !file_path[0]) {
		perr("No output file path specified");
		return false;
	}

	bench_io_start();

	size_t mat_size = (size_t)(g_h5.mat_dim * g_h5.mat_dim);
	size_t bytes = sizeof(*g_h5.matrix) * mat_size;

#ifdef USE_CUDA
	bool device_limited = arg_mode_cuda() && cuda_triangular(bytes);
#else
	bool device_limited = false;
#endif

	const size_t safe = available_memory() * 3 / 4;
	if (device_limited || bytes > safe) {
		mat_size = (size_t)(g_h5.mat_dim * (g_h5.mat_dim - 1) / 2);
		bytes = sizeof(*g_h5.matrix) * mat_size;
		g_h5.mode_mmap = bytes > safe;
		g_h5.triangular = true;
		pverb("Using triangular matrix storage");
	}

	if (g_h5.mode_mmap) {
		char name[MAX_PATH];
		file_matrix_name(name, sizeof(name), file_path);
		pinfo("Matrix size exceeds memory limits");
		if (!file_matrix_open(&g_h5.mmap, name, (size_t)g_h5.mat_dim))
			return false;
		g_h5.matrix = g_h5.mmap.matrix;
		g_h5.matrix_b = g_h5.mmap.meta.bytes;
	} else {
		MALLOC_CL(g_h5.matrix, mat_size);
		if (!g_h5.matrix)
			return false;
		memset(g_h5.matrix, 0, bytes);
		g_h5.matrix_b = bytes;
	}

	pverb("HDF5 matrix size: " Ps64 " x " Ps64, g_h5.mat_dim, g_h5.mat_dim);

	hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);

	g_h5.file_id = H5Fcreate(file_path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
	H5Pclose(fapl);

	if (g_h5.file_id < 0) {
		perr("Failed to create HDF5 file: %s", file_path);
		h5_file_close();
		return false;
	}

	hsize_t matrix_dims[2] = { (hsize_t)g_h5.mat_dim,
				   (hsize_t)g_h5.mat_dim };
	hid_t matrix_space = H5Screate_simple(2, matrix_dims, NULL);

	if (matrix_space < 0) {
		perr("Failed to create matrix dataspace");
		h5_file_close();
		return false;
	}

	hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);

	if (g_h5.mat_dim > H5_MIN_CHUNK_SIZE) {
		s32 chunk_dim = h5_chunk_dimensions_calculate();
		hsize_t chunk_dims[2] = { (hsize_t)chunk_dim,
					  (hsize_t)chunk_dim };
		H5Pset_chunk(plist_id, 2, chunk_dims);
		pverbl("HDF5 chunk size: " Ps32 " x " Ps32, chunk_dim,
		       chunk_dim);

		if (g_h5.compression > 0)
			H5Pset_deflate(plist_id, g_h5.compression);
	}

	g_h5.matrix_id = H5Dcreate2(g_h5.file_id, "/similarity_matrix",
				    H5T_STD_I32LE, matrix_space, H5P_DEFAULT,
				    plist_id, H5P_DEFAULT);

	H5Sclose(matrix_space);
	H5Pclose(plist_id);

	if (g_h5.matrix_id < 0) {
		perr("Failed to create similarity matrix dataset");
		h5_file_close();
		return false;
	}

	pverb("Storing " Ps32 " sequences in HDF5 file", seq_n);

	hid_t seq_group = H5Gcreate2(g_h5.file_id, "/sequences", H5P_DEFAULT,
				     H5P_DEFAULT, H5P_DEFAULT);

	if (seq_group < 0) {
		perr("Failed to create sequences group");
		h5_file_close();
		return false;
	}

	H5Gclose(seq_group);

	hsize_t seq_dims[1] = { (hsize_t)g_h5.mat_dim };
	hid_t lengths_space = H5Screate_simple(1, seq_dims, NULL);

	if (lengths_space < 0) {
		perr("Failed to create sequence lengths dataspace");
		h5_file_close();
		return false;
	}

	g_h5.lengths_id = H5Dcreate2(g_h5.file_id, "/sequences/lengths",
				     H5T_STD_I32LE, lengths_space, H5P_DEFAULT,
				     H5P_DEFAULT, H5P_DEFAULT);

	H5Sclose(lengths_space);

	if (g_h5.lengths_id < 0) {
		perr("Failed to create sequence lengths dataset");
		h5_file_close();
		return false;
	}

	s32 *MALLOC(lengths, (size_t)seq_n);
	if (!lengths) {
		perr("Failed to allocate memory for sequence lengths");
		h5_file_close();
		return false;
	}

	for (s32 i = 0; i < seq_n; i++)
		lengths[i] = seqs[i].length;

	herr_t status = H5Dwrite(g_h5.lengths_id, H5T_NATIVE_INT, H5S_ALL,
				 H5S_ALL, H5P_DEFAULT, lengths);

	free(lengths);

	if (status < 0) {
		perr("Failed to write sequence lengths");
		h5_file_close();
		return false;
	}

	hid_t string_type = H5Tcopy(H5T_C_S1);
	H5Tset_size(string_type, H5T_VARIABLE);

	hid_t seq_space = H5Screate_simple(1, seq_dims, NULL);
	if (seq_space < 0) {
		perr("Failed to create sequences dataspace");
		H5Tclose(string_type);
		h5_file_close();
		return false;
	}

	g_h5.sequences_id = H5Dcreate2(g_h5.file_id, "/sequences/dataset",
				       string_type, seq_space, H5P_DEFAULT,
				       H5P_DEFAULT, H5P_DEFAULT);

	if (g_h5.sequences_id < 0) {
		perr("Failed to create sequences dataset");
		H5Sclose(seq_space);
		H5Tclose(string_type);
		h5_file_close();
		return false;
	}

	ppercent(0, "Storing sequences");

	const s32 batch_size = H5_SEQUENCE_BATCH_SIZE;
	for (s32 batch_start = 0; batch_start < seq_n;
	     batch_start += batch_size) {
		s32 batch_end = min(batch_start + batch_size, seq_n);
		s32 current_batch = batch_end - batch_start;

		char **MALLOC(seq_data, (size_t)current_batch);
		if (!seq_data) {
			perr("Failed to allocate memory for sequence batch");
			H5Sclose(seq_space);
			H5Tclose(string_type);
			h5_file_close();
			return false;
		}

		for (s32 i = 0; i < current_batch; i++)
			seq_data[i] = seqs[batch_start + i].letters;

		hsize_t batch_dims[1] = { (hsize_t)current_batch };
		hid_t batch_mem_space = H5Screate_simple(1, batch_dims, NULL);

		if (batch_mem_space < 0) {
			perr("Failed to create memory dataspace for sequence batch");
			free(seq_data);
			H5Sclose(seq_space);
			H5Tclose(string_type);
			h5_file_close();
			return false;
		}

		hsize_t start[1] = { (hsize_t)batch_start };
		hsize_t count[1] = { (hsize_t)current_batch };
		status = H5Sselect_hyperslab(seq_space, H5S_SELECT_SET, start,
					     NULL, count, NULL);

		if (status >= 0)
			status = H5Dwrite(g_h5.sequences_id, string_type,
					  batch_mem_space, seq_space,
					  H5P_DEFAULT, seq_data);

		H5Sclose(batch_mem_space);
		free(seq_data);

		if (status < 0) {
			perr("Failed to write sequence batch");
			H5Sclose(seq_space);
			H5Tclose(string_type);
			h5_file_close();
			return false;
		}

		pproport(batch_end / seq_n, "Storing sequences");
	}

	ppercent(100, "Storing sequences");

	H5Sclose(seq_space);
	H5Tclose(string_type);
	bench_io_end();
	g_h5.is_init = true;
	return true;
}

void h5_matrix_column_set(s32 col, const s32 *values)
{
	if (!g_h5.mode_write)
		return;

	if (g_h5.triangular) {
		memcpy(g_h5.matrix + ((s64)col * (col - 1)) / 2, values,
		       sizeof(*g_h5.matrix) * (size_t)col);
	} else {
		for (s32 row = 0; row < col; row++) {
			g_h5.matrix[g_h5.mat_dim * row + col] = values[row];
			g_h5.matrix[g_h5.mat_dim * col + row] = values[row];
		}
	}
}

void h5_checksum_set(s64 checksum)
{
	g_h5.checksum = checksum;
}

s64 h5_checksum(void)
{
	return g_h5.checksum;
}

static void h5_store_checksum(void);
static void h5_flush_matrix(void);

void h5_close(int skip_flush)
{
	if (!g_h5.is_init)
		return;

	psection("Finalizing Results");
	pinfo("Matrix checksum: " Ps64, g_h5.checksum);

	if (g_h5.mode_write) {
		bench_io_start();
		if (!skip_flush) {
			pinfol("Writing results to HDF5");
			perr_context("HDF5");
			h5_store_checksum();
			h5_flush_matrix();
		}

		h5_file_close();
		bench_io_end();
	}

	bench_io_print();
	g_h5.is_init = false;
}

#ifdef USE_CUDA

s32 *h5_matrix_data(void)
{
	return g_h5.matrix;
}

size_t h5_matrix_bytes(void)
{
	return g_h5.matrix_b;
}

#endif

static s32 h5_chunk_dimensions_calculate(void)
{
	const s32 mat_dim = (s32)g_h5.mat_dim;
	s32 chunk_dim;

	if (mat_dim <= H5_MIN_CHUNK_SIZE) {
		chunk_dim = mat_dim;
	} else if (mat_dim > H5_MAX_CHUNK_SIZE) {
		s32 target_chunks = mat_dim > 1 << 15 ? 1 << 4 : 1 << 5;
		chunk_dim = mat_dim / target_chunks;
		chunk_dim = ALIGN_POW2(chunk_dim, H5_MIN_CHUNK_SIZE);
		chunk_dim = max(chunk_dim, H5_MIN_CHUNK_SIZE);
		chunk_dim = min(chunk_dim, H5_MAX_CHUNK_SIZE);
	} else {
		s32 chunk_candidates[] = {
			H5_MIN_CHUNK_SIZE,	H5_MIN_CHUNK_SIZE << 1,
			H5_MIN_CHUNK_SIZE << 2, H5_MIN_CHUNK_SIZE << 3,
			H5_MIN_CHUNK_SIZE << 4, H5_MIN_CHUNK_SIZE << 5,
			H5_MIN_CHUNK_SIZE << 6, H5_MAX_CHUNK_SIZE
		};

		s32 num_candidates = ARRAY_SIZE(chunk_candidates);

		chunk_dim = H5_MIN_CHUNK_SIZE;

		for (s32 i = 0; i < num_candidates; i++) {
			s32 chunk = chunk_candidates[i];
			if (chunk > H5_MAX_CHUNK_SIZE || chunk > mat_dim)
				break;

			chunk_dim = chunk;
			if (chunk * 8 >= mat_dim)
				break;
		}
	}

	return chunk_dim;
}

static void h5_file_close(void)
{
	if (g_h5.mode_mmap) {
		file_matrix_close(&g_h5.mmap);
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

	if (g_h5.lengths_id > 0) {
		H5Dclose(g_h5.lengths_id);
		g_h5.lengths_id = H5I_INVALID_HID;
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
	if (!g_h5.mode_write || g_h5.matrix_id < 0 || g_h5.file_id < 0)
		return;

	htri_t attr_exists = H5Aexists(g_h5.matrix_id, "checksum");
	if (attr_exists > 0)
		H5Adelete(g_h5.matrix_id, "checksum");

	hid_t attr_space = H5Screate(H5S_SCALAR);
	if (attr_space < 0) {
		perr("Failed to create dataspace for checksum attribute");
		return;
	}

	hid_t attr_id = H5Acreate2(g_h5.matrix_id, "checksum", H5T_STD_I64LE,
				   attr_space, H5P_DEFAULT, H5P_DEFAULT);

	if (attr_id < 0) {
		perr("Failed to create checksum attribute");
		H5Sclose(attr_space);
		return;
	}

	herr_t status = H5Awrite(attr_id, H5T_NATIVE_INT64, &g_h5.checksum);
	H5Aclose(attr_id);
	H5Sclose(attr_space);

	if (status < 0)
		perr("Failed to write checksum attribute");
}

static void h5_flush_matrix(void)
{
	if (!g_h5.triangular) {
		herr_t status = H5Dwrite(g_h5.matrix_id, H5T_NATIVE_INT,
					 H5S_ALL, H5S_ALL, H5P_DEFAULT,
					 g_h5.matrix);
		if (status < 0)
			perr("Failed to write matrix data to HDF5");
		return;
	}

	const s32 mat_dim = (s32)g_h5.mat_dim;
	const size_t available_mem = available_memory();
	if (!available_mem) {
		perr("Failed to retrieve available memory");
		return;
	}

	const size_t row_bytes = sizeof(*g_h5.matrix) * (size_t)mat_dim;
	const s32 max_rows = (s32)(available_mem / (4 * row_bytes));
	const s32 chunk_rows = h5_chunk_dimensions_calculate();
	s32 chunk_size = chunk_rows > 4 ? chunk_rows : 4;
	if (chunk_size > max_rows && max_rows > 4)
		chunk_size = max_rows;

	const size_t buffer_mib = (row_bytes * (size_t)chunk_size) / MiB;
	pverb("Using " Ps32 " rows per chunk (%zu MiB buffer)", chunk_size,
	      buffer_mib);

	s32 *buffer = calloc((size_t)chunk_size, row_bytes);
	if (!buffer) {
		pwarn("Failed to allocate buffer of %zu bytes",
		      row_bytes * (size_t)chunk_size);

		chunk_size = 1;
		buffer = calloc((size_t)chunk_size, row_bytes);
		if (!buffer) {
			perr("Cannot allocate even minimal buffer, aborting");
			return;
		}

		pwarn("Using minimal buffer size of 1 row (%zu bytes)",
		      row_bytes);
	}

	hid_t file_space = H5Dget_space(g_h5.matrix_id);
	if (file_space < 0) {
		free(buffer);
		return;
	}

	pinfo("Converting triangular matrix to HDF5 format");
	ppercent(0, "Converting to HDF5");

	for (s32 begin = 0; begin < mat_dim; begin += chunk_size) {
		s32 end = min(begin + chunk_size, mat_dim);

		for (s32 i = begin; i < end; i++) {
			s64 row = (s64)mat_dim * (i - begin);

			for (s32 j = i + 1; j < mat_dim; j++) {
				buffer[row + j] =
					g_h5.matrix[matrix_index(i, j)];
			}

			for (s32 j = 0; j < i; j++) {
				if (j >= begin) {
					buffer[row + j] =
						buffer[(s64)mat_dim *
							       (j - begin) +
						       i];
				} else {
					buffer[row + j] =
						g_h5.matrix[matrix_index(j, i)];
				}
			}
		}

		s32 rows = end - begin;
		hsize_t start[2] = { (hsize_t)begin, 0 };
		hsize_t count[2] = { (hsize_t)rows, (hsize_t)mat_dim };
		H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL,
				    count, NULL);

		hsize_t mem_dims[2] = { (hsize_t)rows, (hsize_t)mat_dim };
		hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);

		if (mem_space < 0) {
			perr("Failed to create memory dataspace for matrix chunk");
			H5Sclose(file_space);
			free(buffer);
			return;
		}

		herr_t status = H5Dwrite(g_h5.matrix_id, H5T_NATIVE_INT,
					 mem_space, file_space, H5P_DEFAULT,
					 buffer);

		H5Sclose(mem_space);

		if (status < 0) {
			perr("Failed to write chunk to HDF5");
			H5Sclose(file_space);
			free(buffer);
			return;
		}

		pproport(end / mat_dim, "Converting to HDF5");
	}

	ppercent(100, "Converting to HDF5");
	H5Sclose(file_space);
	free(buffer);
}

ARG_PARSE_L(compression, 10, u8, (u8), (val < 0 || val > 9),
	    "Compression level must be between 0-9")

static void print_compression(void)
{
	pinfom("Compression: " Pu8, g_h5.compression);
}

ARG_EXTERN(output_path);

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
	.action_weight = 450,
	.help_weight = 500,
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(output_path)),
};
