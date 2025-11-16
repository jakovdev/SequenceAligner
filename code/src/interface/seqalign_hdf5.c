#include "interface/seqalign_hdf5.h"

#include <hdf5.h>
#include <string.h>

#include "bio/types.h"
#include "io/files.h"
#include "system/os.h"
#include "system/memory.h"
#include "util/print.h"

#ifdef USE_CUDA
#include "app/args.h"
#include "host_interface.h"
#endif

static struct {
	hsize_t matrix_dims[2];
	hsize_t chunk_dims[2];
	hsize_t seq_dims[1];
	hid_t file_id;
	hid_t matrix_id;
	hid_t sequences_id;
	hid_t lengths_id;
	const char *file_path;
	s32 *full_matrix;
	size_t full_matrix_b;
	struct FileScoreMatrix memory_map;
	u64 matrix_dim;
	s64 checksum;
	char file_matrix_name[MAX_PATH];
	u8 compression_level;
	bool sequences_stored;
	bool mode_write;
	bool memory_map_required;
	bool is_init;
} g_hdf5 = { 0 };

static void h5_chunk_dimensions_calculate(void);
static bool h5_file_setup(void);

bool h5_open(const char *file_path, u64 mat_dim, u8 compression, bool write)
{
	g_hdf5.matrix_dim = mat_dim;
	g_hdf5.file_id = H5I_INVALID_HID;
	g_hdf5.matrix_id = H5I_INVALID_HID;
	g_hdf5.sequences_id = H5I_INVALID_HID;
	g_hdf5.lengths_id = H5I_INVALID_HID;
	g_hdf5.file_path = file_path;
	g_hdf5.compression_level = compression;
	g_hdf5.mode_write = write;

	if (!g_hdf5.mode_write) {
		g_hdf5.is_init = true;
		return true;
	}

	if (g_hdf5.matrix_dim < SEQUENCE_COUNT_MIN) {
		print(M_NONE, ERR "Matrix size is too small");
		return false;
	}

	const size_t bytes_needed =
		sizeof(*g_hdf5.full_matrix) * mat_dim * mat_dim;
	const size_t safe_memory = available_memory() * 3 / 4;

#ifdef USE_CUDA
	g_hdf5.memory_map_required =
		(bytes_needed > safe_memory) ||
		(args_mode_cuda() && cuda_triangular(bytes_needed));
#else
	g_hdf5.memory_map_required = bytes_needed > safe_memory;
#endif

	if (g_hdf5.memory_map_required) {
		file_matrix_name(g_hdf5.file_matrix_name, MAX_PATH,
				 g_hdf5.file_path);
		print(M_LOC(FIRST), INFO "Matrix size exceeds memory limits");
		if (!file_matrix_open(&g_hdf5.memory_map,
				      g_hdf5.file_matrix_name,
				      g_hdf5.matrix_dim))
			return false;
	} else {
		size_t bytes = sizeof(*g_hdf5.full_matrix) * g_hdf5.matrix_dim *
			       g_hdf5.matrix_dim;

		if (!(g_hdf5.full_matrix = alloc_huge_page(bytes)))
			return false;

		memset(g_hdf5.full_matrix, 0, bytes);
		g_hdf5.full_matrix_b = bytes;
	}

	h5_chunk_dimensions_calculate();

	if (!h5_file_setup())
		return false;

	g_hdf5.is_init = true;
	const char *file_type =
		g_hdf5.memory_map_required ? "Memory-mapped file" : "HDF5 file";
	print(M_LOC(LAST), VERBOSE "%s has matrix size: " Pu64 " x " Pu64,
	      file_type, mat_dim, mat_dim);

	return true;
}

#define H5_SEQUENCE_BATCH_SIZE (1 << 12)

bool h5_sequences_store(sequence_t *sequences, u32 seq_count)
{
	if (!g_hdf5.mode_write || !g_hdf5.is_init || g_hdf5.sequences_stored)
		return true;

	if (g_hdf5.file_id < 0) {
		print(M_NONE,
		      ERR "Cannot store sequences: HDF5 file not initialized");
		return false;
	}

	print(M_NONE, INFO "Storing " Pu32 " sequences in HDF5 file",
	      seq_count);

	hid_t seq_group = H5Gcreate2(g_hdf5.file_id, "/sequences", H5P_DEFAULT,
				     H5P_DEFAULT, H5P_DEFAULT);

	if (seq_group < 0) {
		print(M_NONE, ERR "Failed to create sequences group");
		return false;
	}

	H5Gclose(seq_group);

	g_hdf5.seq_dims[0] = g_hdf5.matrix_dim;
	hid_t lengths_space = H5Screate_simple(1, g_hdf5.seq_dims, NULL);

	if (lengths_space < 0) {
		print(M_NONE,
		      ERR "Failed to create sequence lengths dataspace");
		return false;
	}

	g_hdf5.lengths_id = H5Dcreate2(g_hdf5.file_id, "/sequences/lengths",
				       H5T_STD_U64LE, lengths_space,
				       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	H5Sclose(lengths_space);

	if (g_hdf5.lengths_id < 0) {
		print(M_NONE, ERR "Failed to create sequence lengths dataset");
		return false;
	}

	u64 *lengths = MALLOC(lengths, seq_count);
	if (!lengths) {
		print(M_NONE,
		      ERR "Failed to allocate memory for sequence lengths");
		H5Dclose(g_hdf5.lengths_id);
		g_hdf5.lengths_id = H5I_INVALID_HID;
		return false;
	}

	for (u32 i = 0; i < seq_count; i++)
		lengths[i] = sequences[i].length;

	herr_t status = H5Dwrite(g_hdf5.lengths_id, H5T_NATIVE_ULONG, H5S_ALL,
				 H5S_ALL, H5P_DEFAULT, lengths);

	free(lengths);

	if (status < 0) {
		print(M_NONE, ERR "Failed to write sequence lengths");
		H5Dclose(g_hdf5.lengths_id);
		g_hdf5.lengths_id = H5I_INVALID_HID;
		return false;
	}

	hid_t string_type = H5Tcopy(H5T_C_S1);
	H5Tset_size(string_type, H5T_VARIABLE);

	hid_t seq_space = H5Screate_simple(1, g_hdf5.seq_dims, NULL);
	if (seq_space < 0) {
		print(M_NONE, ERR "Failed to create sequences dataspace");
		H5Tclose(string_type);
		H5Dclose(g_hdf5.lengths_id);
		g_hdf5.lengths_id = H5I_INVALID_HID;
		return false;
	}

	g_hdf5.sequences_id = H5Dcreate2(g_hdf5.file_id, "/sequences/dataset",
					 string_type, seq_space, H5P_DEFAULT,
					 H5P_DEFAULT, H5P_DEFAULT);

	if (g_hdf5.sequences_id < 0) {
		print(M_NONE, ERR "Failed to create sequences dataset");
		H5Sclose(seq_space);
		H5Tclose(string_type);
		H5Dclose(g_hdf5.lengths_id);
		g_hdf5.lengths_id = H5I_INVALID_HID;
		return false;
	}

	print(M_PERCENT(0) "Storing sequences");

	const u32 batch_size = H5_SEQUENCE_BATCH_SIZE;
	for (u32 batch_start = 0; batch_start < seq_count;
	     batch_start += batch_size) {
		u32 batch_end = min(batch_start + batch_size, seq_count);
		u32 current_batch_size = batch_end - batch_start;

		char **seq_data = MALLOC(seq_data, current_batch_size);
		if (!seq_data) {
			print(M_NONE, ERR
			      "Failed to allocate memory for sequence batch");
			H5Sclose(seq_space);
			H5Tclose(string_type);
			H5Dclose(g_hdf5.sequences_id);
			g_hdf5.sequences_id = H5I_INVALID_HID;
			H5Dclose(g_hdf5.lengths_id);
			g_hdf5.lengths_id = H5I_INVALID_HID;
			return false;
		}

		for (u32 i = 0; i < current_batch_size; i++)
			seq_data[i] = sequences[batch_start + i].letters;

		hsize_t batch_dims[1] = { current_batch_size };
		hid_t batch_mem_space = H5Screate_simple(1, batch_dims, NULL);

		if (batch_mem_space < 0) {
			print(M_NONE, ERR
			      "Failed to create memory dataspace for sequence batch");
			free(seq_data);
			H5Sclose(seq_space);
			H5Tclose(string_type);
			H5Dclose(g_hdf5.sequences_id);
			g_hdf5.sequences_id = H5I_INVALID_HID;
			H5Dclose(g_hdf5.lengths_id);
			g_hdf5.lengths_id = H5I_INVALID_HID;
			return false;
		}

		hsize_t start[1] = { batch_start };
		hsize_t count[1] = { current_batch_size };
		status = H5Sselect_hyperslab(seq_space, H5S_SELECT_SET, start,
					     NULL, count, NULL);

		if (status >= 0)
			status = H5Dwrite(g_hdf5.sequences_id, string_type,
					  batch_mem_space, seq_space,
					  H5P_DEFAULT, seq_data);

		H5Sclose(batch_mem_space);
		free(seq_data);

		if (status < 0) {
			print(M_NONE, ERR "Failed to write sequence batch");
			H5Sclose(seq_space);
			H5Tclose(string_type);
			H5Dclose(g_hdf5.sequences_id);
			g_hdf5.sequences_id = H5I_INVALID_HID;
			H5Dclose(g_hdf5.lengths_id);
			g_hdf5.lengths_id = H5I_INVALID_HID;
			return false;
		}

		print(M_PROPORT(batch_end / seq_count) "Storing sequences");
	}

	print(M_PERCENT(100) "Storing sequences");

	H5Sclose(seq_space);
	H5Tclose(string_type);
	g_hdf5.sequences_stored = true;
	return true;
}

void h5_matrix_set(u32 row, u32 col, s32 value)
{
	if (!g_hdf5.mode_write)
		return;

	if (g_hdf5.memory_map_required) {
		g_hdf5.memory_map.matrix[matrix_index(row, col)] = value;
	} else {
		g_hdf5.full_matrix[g_hdf5.matrix_dim * row + col] = value;
		g_hdf5.full_matrix[g_hdf5.matrix_dim * col + row] = value;
	}
}

void h5_checksum_set(s64 checksum)
{
	g_hdf5.checksum = checksum;
}

s64 h5_checksum(void)
{
	return g_hdf5.checksum;
}

static void h5_store_checksum(void);
static void h5_flush_memory_map(void);
static void h5_flush_full_matrix(void);
static void h5_file_close(void);

void h5_close(int skip_flush)
{
	if (!g_hdf5.is_init)
		return;

	if (g_hdf5.mode_write) {
		if (!skip_flush) {
			print(M_NONE, SECTION "Finalizing Results");
			print(M_LOC(FIRST), INFO "Matrix checksum: " Ps64,
			      g_hdf5.checksum);
			print(M_LOC(LAST), INFO "Writing results to %s",
			      file_name_path(g_hdf5.file_path));
			print_error_context("HDF5");
			h5_store_checksum();
			g_hdf5.memory_map_required ? h5_flush_memory_map() :
						     h5_flush_full_matrix();
		}

		h5_file_close();
	}

	g_hdf5.is_init = false;
}

#ifdef USE_CUDA

s32 *h5_matrix_data(void)
{
	if (!g_hdf5.mode_write)
		return NULL;

	return g_hdf5.memory_map_required ? g_hdf5.memory_map.matrix :
					    g_hdf5.full_matrix;
}

size_t h5_matrix_bytes(void)
{
	if (!g_hdf5.mode_write)
		return 0;

	return g_hdf5.memory_map_required ? g_hdf5.memory_map.meta.bytes :
					    g_hdf5.full_matrix_b;
}

#endif

#define H5_MIN_CHUNK_SIZE (1 << 7)
#define H5_MAX_CHUNK_SIZE (H5_MIN_CHUNK_SIZE << 7)

static void h5_chunk_dimensions_calculate(void)
{
	u32 matrix_dim = (u32)g_hdf5.matrix_dim;
	u32 chunk_dim;

	if (matrix_dim <= H5_MIN_CHUNK_SIZE) {
		chunk_dim = matrix_dim;
	} else if (matrix_dim > H5_MAX_CHUNK_SIZE) {
		u32 target_chunks = matrix_dim > 1 << 15 ? 1 << 4 : 1 << 5;
		chunk_dim = matrix_dim / target_chunks;
		chunk_dim = ALIGN_POW2(chunk_dim, H5_MIN_CHUNK_SIZE);
		chunk_dim = max(chunk_dim, H5_MIN_CHUNK_SIZE);
		chunk_dim = min(chunk_dim, H5_MAX_CHUNK_SIZE);
	} else {
		u32 chunk_candidates[] = {
			H5_MIN_CHUNK_SIZE,	H5_MIN_CHUNK_SIZE << 1,
			H5_MIN_CHUNK_SIZE << 2, H5_MIN_CHUNK_SIZE << 3,
			H5_MIN_CHUNK_SIZE << 4, H5_MIN_CHUNK_SIZE << 5,
			H5_MIN_CHUNK_SIZE << 6, H5_MAX_CHUNK_SIZE
		};

		u32 num_candidates = ARRAY_SIZE(chunk_candidates);

		chunk_dim = H5_MIN_CHUNK_SIZE;

		for (u32 i = 0; i < num_candidates; i++) {
			u32 candidate = chunk_candidates[i];
			if (candidate > H5_MAX_CHUNK_SIZE ||
			    candidate > matrix_dim) {
				break;
			}

			chunk_dim = candidate;
			if (candidate * 8 >= matrix_dim)
				break;
		}
	}

	g_hdf5.chunk_dims[0] = chunk_dim;
	g_hdf5.chunk_dims[1] = chunk_dim;
	print(M_LOC(FIRST), VERBOSE "HDF5 chunk size: " Pu32 " x " Pu32,
	      chunk_dim, chunk_dim);
}

static bool h5_file_setup(void)
{
	if (!g_hdf5.mode_write)
		return true;

	hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);

	g_hdf5.file_id =
		H5Fcreate(g_hdf5.file_path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
	H5Pclose(fapl);

	if (g_hdf5.file_id < 0) {
		print(M_NONE, ERR "Failed to create HDF5 file: %s",
		      g_hdf5.file_path);
		h5_file_close();
		return false;
	}

	g_hdf5.matrix_dims[0] = g_hdf5.matrix_dim;
	g_hdf5.matrix_dims[1] = g_hdf5.matrix_dim;
	hid_t matrix_space = H5Screate_simple(2, g_hdf5.matrix_dims, NULL);

	if (matrix_space < 0) {
		print(M_NONE, ERR "Failed to create matrix dataspace");
		h5_file_close();
		return false;
	}

	hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);

	if (g_hdf5.matrix_dim > H5_MIN_CHUNK_SIZE) {
		H5Pset_chunk(plist_id, 2, g_hdf5.chunk_dims);

		if (g_hdf5.compression_level > 0)
			H5Pset_deflate(plist_id, g_hdf5.compression_level);
	}

	g_hdf5.matrix_id = H5Dcreate2(g_hdf5.file_id, "/similarity_matrix",
				      H5T_STD_I32LE, matrix_space, H5P_DEFAULT,
				      plist_id, H5P_DEFAULT);

	H5Sclose(matrix_space);
	H5Pclose(plist_id);

	if (g_hdf5.matrix_id < 0) {
		print(M_NONE, ERR "Failed to create similarity matrix dataset");
		h5_file_close();
		return false;
	}

	return true;
}

static void h5_file_close(void)
{
	if (g_hdf5.memory_map_required) {
		file_matrix_close(&g_hdf5.memory_map);
		remove(g_hdf5.file_matrix_name);
	} else {
		if (g_hdf5.full_matrix) {
			aligned_free(g_hdf5.full_matrix);
			g_hdf5.full_matrix = NULL;
		}

		g_hdf5.full_matrix_b = 0;
	}

	if (g_hdf5.sequences_id > 0) {
		H5Dclose(g_hdf5.sequences_id);
		g_hdf5.sequences_id = H5I_INVALID_HID;
	}

	if (g_hdf5.lengths_id > 0) {
		H5Dclose(g_hdf5.lengths_id);
		g_hdf5.lengths_id = H5I_INVALID_HID;
	}

	if (g_hdf5.matrix_id > 0) {
		H5Dclose(g_hdf5.matrix_id);
		g_hdf5.matrix_id = H5I_INVALID_HID;
	}

	if (g_hdf5.file_id > 0) {
		H5Fclose(g_hdf5.file_id);
		g_hdf5.file_id = H5I_INVALID_HID;
	}
}

static void h5_store_checksum(void)
{
	if (!g_hdf5.mode_write || g_hdf5.matrix_id < 0 || g_hdf5.file_id < 0)
		return;

	htri_t attr_exists = H5Aexists(g_hdf5.matrix_id, "checksum");
	if (attr_exists > 0)
		H5Adelete(g_hdf5.matrix_id, "checksum");

	hid_t attr_space = H5Screate(H5S_SCALAR);
	if (attr_space < 0) {
		print(M_NONE,
		      ERR "Failed to create dataspace for checksum attribute");
		return;
	}

	hid_t attr_id = H5Acreate2(g_hdf5.matrix_id, "checksum", H5T_STD_I64LE,
				   attr_space, H5P_DEFAULT, H5P_DEFAULT);

	if (attr_id < 0) {
		print(M_NONE, ERR "Failed to create checksum attribute");
		H5Sclose(attr_space);
		return;
	}

	herr_t status = H5Awrite(attr_id, H5T_NATIVE_INT64, &g_hdf5.checksum);
	H5Aclose(attr_id);
	H5Sclose(attr_space);

	if (status < 0) {
		print(M_NONE, ERR "Failed to write checksum attribute");
		return;
	}

	return;
}

static void h5_flush_full_matrix(void)
{
	herr_t status = H5Dwrite(g_hdf5.matrix_id, H5T_NATIVE_INT, H5S_ALL,
				 H5S_ALL, H5P_DEFAULT, g_hdf5.full_matrix);

	if (status < 0) {
		print(M_NONE, ERR "Failed to write matrix data to HDF5");
		return;
	}

	return;
}

static void h5_flush_memory_map(void)
{
	size_t matrix_dim = g_hdf5.matrix_dim;
	size_t available_mem = available_memory();
	if (!available_mem) {
		print(M_NONE, ERR "Failed to retrieve available memory");
		return;
	}

	hsize_t chunk_rows = g_hdf5.chunk_dims[0];
	size_t row_bytes = matrix_dim * sizeof(*g_hdf5.memory_map.matrix);
	u32 max_rows = (u32)(available_mem / (4 * row_bytes));
	u32 chunk_size = (u32)(chunk_rows > 4 ? chunk_rows : 4);
	if (chunk_size > max_rows && max_rows > 4)
		chunk_size = max_rows;

	const size_t buffer_mib = (chunk_size * row_bytes) / MiB;
	print(M_NONE, VERBOSE "Using " Pu32 " rows per chunk (%zu MiB buffer)",
	      chunk_size, buffer_mib);

	s32 *buffer = calloc(chunk_size, row_bytes);
	if (!buffer) {
		print(M_NONE, WARNING "Failed to allocate buffer of %zu bytes",
		      row_bytes * chunk_size);

		chunk_size = 1;
		buffer = calloc(chunk_size, row_bytes);
		if (!buffer) {
			print(M_NONE, ERR
			      "Cannot allocate even minimal buffer, aborting");
			return;
		}

		print(M_NONE,
		      WARNING "Using minimal buffer size of 1 row (%zu bytes)",
		      row_bytes);
	}

	hid_t file_space = H5Dget_space(g_hdf5.matrix_id);
	if (file_space < 0) {
		free(buffer);
		return;
	}

	print(M_NONE, INFO "Converting memory-mapped matrix to HDF5 format");
	print(M_PERCENT(0) "Converting to HDF5");

	for (u32 begin = 0; begin < matrix_dim; begin += chunk_size) {
		u32 end = min(begin + chunk_size, (u32)matrix_dim);

		for (u32 i = begin; i < end; i++) {
			u64 row = matrix_dim * (i - begin);

			for (u32 j = i + 1; j < matrix_dim; j++) {
				buffer[row + j] =
					g_hdf5.memory_map
						.matrix[matrix_index(i, j)];
			}

			for (u32 j = 0; j < i; j++) {
				if (j >= begin) {
					buffer[row + j] =
						buffer[matrix_dim * (j - begin) +
						       i];
				} else {
					buffer[row + j] =
						g_hdf5.memory_map
							.matrix[matrix_index(
								j, i)];
				}
			}
		}

		u32 rows = end - begin;
		hsize_t start[2] = { begin, 0 };
		hsize_t count[2] = { rows, matrix_dim };
		H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL,
				    count, NULL);

		hsize_t mem_dims[2] = { rows, matrix_dim };
		hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);

		if (mem_space < 0) {
			print(M_NONE, ERR
			      "Failed to create memory dataspace for matrix chunk");
			H5Sclose(file_space);
			free(buffer);
			return;
		}

		herr_t status = H5Dwrite(g_hdf5.matrix_id, H5T_NATIVE_INT,
					 mem_space, file_space, H5P_DEFAULT,
					 buffer);

		H5Sclose(mem_space);

		if (status < 0) {
			print(M_NONE, ERR "Failed to write chunk to HDF5");
			H5Sclose(file_space);
			free(buffer);
			return;
		}

		print(M_PROPORT(end / matrix_dim) "Converting to HDF5");
	}

	print(M_PERCENT(100) "Converting to HDF5");
	H5Sclose(file_space);
	free(buffer);
	return;
}
