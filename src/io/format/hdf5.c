#include <args.h>
#include <hdf5.h>
#include <print.h>

#include "io/output.h"
#include "system/os.h"
#include "system/memory.h"
#include "util/macros.h"

#define H5_MAX_CHUNK_SIZE PAGE_SIZE
#define H5_MIN_CHUNK_SIZE (1 << 8)
unsigned int COMPRESSION;

static bool flush_hdf5(struct output *sm, const char *path)
{
	hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
	H5Pset_alignment(fapl, H5_MAX_CHUNK_SIZE, H5_MAX_CHUNK_SIZE);
	hid_t file_id = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
	H5Pclose(fapl);
	if (file_id < 0) {
		perr("Failed to create HDF5 file: %s", file_name(path));
		return false;
	}

	pinfo("Writing %zu sequences to HDF5", sm->dim);

	hsize_t seq_dims[1] = { sm->dim };
	hid_t seq_space = H5Screate_simple(1, seq_dims, nullptr);
	if (seq_space < 0) {
		perr("Failed to create HDF5 dataspace for sequences");
		H5Fclose(file_id);
		return false;
	}

	hid_t string_type = H5Tcopy(H5T_C_S1);
	H5Tset_size(string_type, H5T_VARIABLE);
	hid_t sequences_id = H5Dcreate2(file_id, "/sequences", string_type,
					seq_space, H5P_DEFAULT, H5P_DEFAULT,
					H5P_DEFAULT);
	if (sequences_id < 0) {
		perr("Failed to create HDF5 dataset for sequences");
		H5Sclose(seq_space);
		H5Tclose(string_type);
		H5Fclose(file_id);
		return false;
	}

	herr_t status = H5Dwrite(sequences_id, string_type, H5S_ALL, H5S_ALL,
				 H5P_DEFAULT, sm->seqs);
	H5Dclose(sequences_id);
	H5Sclose(seq_space);
	H5Tclose(string_type);
	if (status < 0) {
		perr("Failed to write sequence data to HDF5 dataset");
		H5Fclose(file_id);
		return false;
	}

	hsize_t matrix_dims[2] = { sm->dim, sm->dim };
	hid_t matrix_space = H5Screate_simple(2, matrix_dims, nullptr);
	if (matrix_space < 0) {
		perr("Failed to create HDF5 dataspace for Similarity Matrix");
		H5Fclose(file_id);
		return false;
	}

	hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);

	size_t chunk_dim = sm->dim;
	if (sm->dim > H5_MIN_CHUNK_SIZE) {
		chunk_dim = 64;
		size_t square =
			(size_t)(chunk_dim * chunk_dim) * sizeof(chunk_dim);
		size_t target_bytes = (2 * MiB) / (1 + COMPRESSION / 3);
		while (chunk_dim < sm->dim && square < target_bytes)
			chunk_dim *= 2;
		if (chunk_dim > sm->dim || square > target_bytes)
			chunk_dim /= 2;

		chunk_dim = max(chunk_dim, H5_MIN_CHUNK_SIZE);
		chunk_dim = min(chunk_dim, H5_MAX_CHUNK_SIZE);
		chunk_dim = min(chunk_dim, sm->dim);
		hsize_t chunk_dims[2] = { chunk_dim, chunk_dim };
		H5Pset_chunk(plist_id, 2, chunk_dims);
		pverb("HDF5 chunk size: %zu x %zu", chunk_dim, chunk_dim);

		if (COMPRESSION)
			H5Pset_deflate(plist_id, COMPRESSION);
	}
	hid_t matrix_id = H5Dcreate2(file_id, "/similarity_matrix",
				     H5T_STD_I32LE, matrix_space, H5P_DEFAULT,
				     plist_id, H5P_DEFAULT);
	H5Pclose(plist_id);
	H5Sclose(matrix_space);
	if (matrix_id < 0) {
		perr("Failed to create HDF5 dataset for Similarity Matrix");
		H5Fclose(file_id);
		return false;
	}

	hid_t attr_space = H5Screate(H5S_SCALAR);
	if (attr_space < 0) {
		perr("Failed to create HDF5 dataspace for checksum attribute");
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		return false;
	}

	hid_t attr_id = H5Acreate2(matrix_id, "checksum", H5T_STD_I64LE,
				   attr_space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr_id < 0) {
		perr("Failed to create checksum attribute");
		H5Sclose(attr_space);
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		return false;
	}

	status = H5Awrite(attr_id, H5T_NATIVE_INT64, &sm->checksum);
	H5Aclose(attr_id);
	H5Sclose(attr_space);
	if (status < 0) {
		perr("Failed to write checksum attribute");
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		return false;
	}

	if (!sm->triangular) {
		pinfo("Writing Similarity Matrix to HDF5");
		herr_t status = H5Dwrite(matrix_id, H5T_NATIVE_INT32, H5S_ALL,
					 H5S_ALL, H5P_DEFAULT, sm->matrix);
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		if (status < 0) {
			perr("Failed to write Similarity Matrix to HDF5");
			return false;
		}
		return true;
	}

	pinfo("Writing triangular Similarity Matrix to HDF5");

	size_t available = available_memory();
	if (!available) {
		perr("Failed to retrieve available memory");
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		return false;
	}

	s64 dim = sm->dim;
	size_t row_bytes = bytesof(sm->matrix, sm->dim);
	s32 max_rows = (s32)(available / (4 * row_bytes));
	s32 chunk_size = (s32)(chunk_dim > 4 ? chunk_dim : 4);
	if (chunk_size > max_rows && max_rows > 4)
		chunk_size = max_rows;

	s32 *MALLOC_AL(buf, PAGE_SIZE, row_bytes * (size_t)chunk_size);
	if (!buf) {
		pwarn("Out of memory, trying minimal amount for conversion");
		chunk_size = 1;
		MALLOC_AL(buf, CACHE_LINE, row_bytes * (size_t)chunk_size);
		if (!buf) {
			perr("Out of memory, aborting conversion");
			H5Dclose(matrix_id);
			H5Fclose(file_id);
			return false;
		}
	}
	memset(buf, 0, row_bytes * (size_t)chunk_size);

	hid_t file_space = H5Dget_space(matrix_id);
	if (file_space < 0) {
		perr("Failed to get HDF5 dataspace for Similarity Matrix");
		free_aligned(buf);
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		return false;
	}

	ppercent(0, "Converting to HDF5");
#define tridx(row, col) (((s64)(col) * ((col) - 1)) / 2 + (row))
	for (s32 off = 0; off < dim; off += chunk_size) {
		s32 end = min(off + chunk_size, dim);
		for (s32 i = off; i < end; i++) {
			s64 row = dim * (i - off);
			for (s32 j = i + 1; j < dim; j++)
				buf[row + j] = sm->matrix[tridx(i, j)];
			for (s32 j = 0; j < i; j++) {
				if (j >= off)
					buf[row + j] = buf[dim * (j - off) + i];
				else
					buf[row + j] = sm->matrix[tridx(j, i)];
			}
		}

		s32 rows = end - off;
		hsize_t start[2] = { (hsize_t)off, 0 };
		hsize_t count[2] = { (hsize_t)rows, sm->dim };
		H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr,
				    count, nullptr);

		hsize_t mem_dims[2] = { (hsize_t)rows, sm->dim };
		hid_t mem_space = H5Screate_simple(2, mem_dims, nullptr);
		if (mem_space < 0) {
			perr("Failed to create memory dataspace for matrix chunk");
			free_aligned(buf);
			H5Sclose(file_space);
			H5Dclose(matrix_id);
			H5Fclose(file_id);
			return false;
		}

		herr_t status = H5Dwrite(matrix_id, H5T_NATIVE_INT32, mem_space,
					 file_space, H5P_DEFAULT, buf);
		H5Sclose(mem_space);
		if (status < 0) {
			perr("Failed to write chunk to HDF5");
			free_aligned(buf);
			H5Sclose(file_space);
			H5Dclose(matrix_id);
			H5Fclose(file_id);
			return false;
		}

		pproport(end / sm->dim, "Converting to HDF5");
	}

	ppercent(100, "Converting to HDF5");
	free_aligned(buf);
	H5Sclose(file_space);
	H5Dclose(matrix_id);
	H5Fclose(file_id);
	return true;
}
FLUSH_REGISTER(FLUSH_HDF5, flush_hdf5)

ARG_PARSE_UL(compression, 10, unsigned int, (unsigned int), val > 9,
	     "Compression level must be between 0-9")

static void print_compression(void)
{
	pinfom("Compression: %u", COMPRESSION);
}

ARG_EXTERN(filter_threshold);
ARG_EXTERN(output_path);

ARGUMENT(compression) = {
	.opt = 'z',
	.lopt = "compression",
	.help = "Compression level for HDF5 datasets [0-9]",
	.param = "N",
	.param_req = ARG_PARAM_REQUIRED,
	.dest = &COMPRESSION,
	.parse_callback = parse_compression,
	.action_callback = print_compression,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(filter_threshold)),
	.help_order = ARG_ORDER_AFTER(ARG(filter_threshold)),
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(output_path)),
};
