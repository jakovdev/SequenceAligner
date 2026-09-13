#include "io/writer.h"

#include <args.h>
#include <hdf5.h>
#include <print.h>
#include <string.h>
#include <strings.h>

#include "bio/method.h"
#include "system/os.h"
#include "util/benchmark.h" /* TEMP */
#include "util/macros.h"

constexpr usz H5_MAX_CHUNK_SIZE = 4 * KiB;
constexpr usz H5_MIN_CHUNK_SIZE = 1 * KiB / 4;
unsigned int COMPRESSION;

static bool hdf5_ext(const char *path)
{
	static const char *EXTS[] = { "hdf5", "h5", nullptr };
	const char *name = file_name(path);
	const char *dot = strrchr(name, '.');
	if (dot && dot != name) {
		for (const char **ext = EXTS; *ext; ext++) {
			if (strcasecmp(*ext, dot + 1) == 0)
				return true;
		}
	}
	return print_Yn("Invalid file extension, default to hdf5?"); /* TEMP */
}

static enum writer_result write_hdf5(struct output out, struct input in,
				     const char *path)
{
	pverbm("Trying out HDF5 writer");
	if (!hdf5_ext(path))
		return WRITER_UNSUPPORTED;
	pverbl("Using HDF5 writer");
	bench_output_start(); /* TEMP */
	hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
	H5Pset_alignment(fapl, H5_MAX_CHUNK_SIZE, H5_MAX_CHUNK_SIZE);
	hid_t file_id = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
	H5Pclose(fapl);
	if (file_id < 0) {
		perr("Failed to create HDF5 file: %s", file_name(path));
		return WRITER_ERROR;
	}

	usz dim = in.num;
	pinfo("Writing %zu sequences to HDF5", dim);

	hsize_t seq_dims[1] = { dim };
	hid_t seq_space = H5Screate_simple(1, seq_dims, nullptr);
	if (seq_space < 0) {
		perr("Failed to create HDF5 dataspace for sequences");
		H5Fclose(file_id);
		return WRITER_ERROR;
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
		return WRITER_ERROR;
	}

	const char **MALLOCA(seqs, dim);
	if (!seqs) {
		perr("Out of memory allocating HDF5 sequence data");
		H5Dclose(sequences_id);
		H5Sclose(seq_space);
		H5Tclose(string_type);
		H5Fclose(file_id);
		return WRITER_ERROR;
	}

	for (uhz i = 0; i < in.num; i++)
		seqs[i] = (const char *)(in.seqs + in.meta[i].off);

	herr_t status = H5Dwrite(sequences_id, string_type, H5S_ALL, H5S_ALL,
				 H5P_DEFAULT, seqs);
	free(seqs);
	H5Dclose(sequences_id);
	H5Sclose(seq_space);
	H5Tclose(string_type);
	if (status < 0) {
		perr("Failed to write sequence data to HDF5 dataset");
		H5Fclose(file_id);
		return WRITER_ERROR;
	}

	hsize_t matrix_dims[2] = { dim, dim };
	hid_t matrix_space = H5Screate_simple(2, matrix_dims, nullptr);
	if (matrix_space < 0) {
		perr("Failed to create HDF5 dataspace for Similarity Matrix");
		H5Fclose(file_id);
		return WRITER_ERROR;
	}

	hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);

	usz chunk_dim = dim;
	if (chunk_dim > H5_MIN_CHUNK_SIZE) {
		chunk_dim = 64;
		usz square = chunk_dim * chunk_dim * sizeof(chunk_dim);
		usz target_bytes = (2 * MiB) / (1 + COMPRESSION / 3);
		while (chunk_dim < dim && square < target_bytes)
			chunk_dim *= 2;
		if (chunk_dim > dim || square > target_bytes)
			chunk_dim /= 2;

		chunk_dim = max(chunk_dim, H5_MIN_CHUNK_SIZE);
		chunk_dim = min(chunk_dim, H5_MAX_CHUNK_SIZE);
		chunk_dim = min(chunk_dim, dim);
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
		return WRITER_ERROR;
	}

	if (!out.triangular) {
		pinfo("Writing Similarity Matrix to HDF5");
		status = H5Dwrite(matrix_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL,
				  H5P_DEFAULT, out.matrix);
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		if (status < 0) {
			perr("Failed to write Similarity Matrix to HDF5");
			return WRITER_ERROR;
		}
		pverb("HDF5 writing finished successfuly");
		return WRITER_SUCCESS;
	}

	pinfo("Writing triangular Similarity Matrix to HDF5");

	usz available = memory_cpu();
	if (!available) {
		perr("Failed to retrieve available memory");
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		return WRITER_ERROR;
	}

	usz row_bytes = bytesof(out.matrix, dim);
	uhz max_rows = available / (4 * row_bytes);
	uhz chunk_size = max(chunk_dim, 4);
	if (chunk_size > max_rows && max_rows > 4)
		chunk_size = max_rows;

	shz *buf = alloc_mmap(row_bytes * chunk_size, false);
	if (!buf) {
		perr("Out of memory during HDF5 conversion");
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		return WRITER_ERROR;
	}

	hid_t file_space = H5Dget_space(matrix_id);
	if (file_space < 0) {
		perr("Failed to get HDF5 dataspace for Similarity Matrix");
		free_mmap(buf);
		H5Dclose(matrix_id);
		H5Fclose(file_id);
		return WRITER_ERROR;
	}

	ppercent(0, "Converting to HDF5");
#define tridx(row, col) (alignments((usz)(col)) + (row))
	for (uhz off = 0; off < dim; off += chunk_size) {
		uhz end = min(off + chunk_size, dim);
		for (uhz i = off; i < end; i++) {
			usz row = dim * (i - off);
			for (uhz j = i + 1; j < dim; j++)
				buf[row + j] = out.matrix[tridx(i, j)];
			for (uhz j = 0; j < i; j++) {
				if (j >= off)
					buf[row + j] = buf[dim * (j - off) + i];
				else
					buf[row + j] = out.matrix[tridx(j, i)];
			}
		}

		uhz rows = end - off;
		hsize_t start[2] = { off, 0 };
		hsize_t count[2] = { rows, dim };
		H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr,
				    count, nullptr);

		hsize_t mem_dims[2] = { rows, dim };
		hid_t mem_space = H5Screate_simple(2, mem_dims, nullptr);
		if (mem_space < 0) {
			perr("Failed to create memory dataspace for matrix chunk");
			free_mmap(buf);
			H5Sclose(file_space);
			H5Dclose(matrix_id);
			H5Fclose(file_id);
			return WRITER_ERROR;
		}

		status = H5Dwrite(matrix_id, H5T_NATIVE_INT32, mem_space,
				  file_space, H5P_DEFAULT, buf);
		H5Sclose(mem_space);
		if (status < 0) {
			perr("Failed to write chunk to HDF5");
			free_mmap(buf);
			H5Sclose(file_space);
			H5Dclose(matrix_id);
			H5Fclose(file_id);
			return WRITER_ERROR;
		}

		pproport(end / dim, "Converting to HDF5");
	}

	ppercent(100, "Converting to HDF5");
	free_mmap(buf);
	H5Sclose(file_space);
	H5Dclose(matrix_id);
	H5Fclose(file_id);
	pverb("HDF5 writing finished successfuly");
	return WRITER_SUCCESS;
}
WRITER_REGISTER(hdf5, write_hdf5);

ARG_PARSE_UL(parse_compression, 10, unsigned int, (unsigned int), val > 9,
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

#ifdef MINGW_LIBSZ
int SZ_BufftoBuffCompress(void *, usz *, const void *, usz, void *);
int SZ_BufftoBuffDecompress(void *, usz *, const void *, usz, void *);
int SZ_encoder_enabled(void);
void *__imp_SZ_BufftoBuffCompress = (void *)SZ_BufftoBuffCompress;
void *__imp_SZ_BufftoBuffDecompress = (void *)SZ_BufftoBuffDecompress;
void *__imp_SZ_encoder_enabled = (void *)SZ_encoder_enabled;
#endif
