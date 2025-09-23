#include "seqalign_hdf5.h"

#include "arch.h"
#include "biotypes.h"
#include "files.h"
#include "print.h"

#include <hdf5.h>

#ifdef USE_CUDA
#include "host_interface.h"
#endif

static struct
{
    char file_matrix_name[MAX_PATH];

    hsize_t matrix_dims[2];
    hsize_t chunk_dims[2];
    hsize_t seq_dims[1];
    hid_t file_id;
    hid_t matrix_id;
    hid_t sequences_id;
    hid_t lengths_id;

    const char* file_path;

#ifdef USE_CUDA
    size_t* triangle_indices_64;
    half_t* triangle_indices_32;
#endif

    score_t* full_matrix;
    size_t full_matrix_b;

    FileScoreMatrix memory_map;

    sequence_count_t matrix_dim;

    int64_t checksum;

    unsigned int compression_level;
    bool sequences_stored;
    bool mode_write;
    bool memory_map_required;
    bool is_init;
} g_hdf5 = { 0 };

static void h5_chunk_dimensions_calculate(void);
static bool h5_file_setup(void);

bool
h5_open(const char* file_path, sequence_count_t mat_dim, unsigned int compression, bool write)
{
    g_hdf5.matrix_dim = mat_dim;
    g_hdf5.file_id = H5I_INVALID_HID;
    g_hdf5.matrix_id = H5I_INVALID_HID;
    g_hdf5.sequences_id = H5I_INVALID_HID;
    g_hdf5.lengths_id = H5I_INVALID_HID;
    g_hdf5.file_path = file_path;
    g_hdf5.compression_level = compression;
    g_hdf5.mode_write = write;

    if (!g_hdf5.mode_write)
    {
        g_hdf5.is_init = true;
        return true;
    }

    if (g_hdf5.matrix_dim < 1)
    {
        print(ERROR, MSG_NONE, "Matrix size is too small");
        return false;
    }

    const size_t bytes_needed = sizeof(*g_hdf5.full_matrix) * mat_dim * mat_dim;
    const size_t safe_memory = available_memory() * 3 / 4;

#ifdef USE_CUDA
    g_hdf5.memory_map_required = (bytes_needed > safe_memory) || cuda_triangular(bytes_needed);
#else
    g_hdf5.memory_map_required = bytes_needed > safe_memory;
#endif

    if (g_hdf5.memory_map_required)
    {
        file_matrix_name(g_hdf5.file_matrix_name, MAX_PATH, g_hdf5.file_path);
        print(INFO, MSG_LOC(FIRST), "Matrix size exceeds RAM threshold, using memory-mapping");
        g_hdf5.memory_map = file_matrix_open(g_hdf5.file_matrix_name, g_hdf5.matrix_dim);
        if (!g_hdf5.memory_map.matrix)
        {
            return false;
        }
    }

    else
    {
        size_t bytes = sizeof(*g_hdf5.full_matrix) * g_hdf5.matrix_dim * g_hdf5.matrix_dim;

        if (!(g_hdf5.full_matrix = CAST(g_hdf5.full_matrix)(alloc_huge_page(bytes))))
        {
            return false;
        }

        memset(g_hdf5.full_matrix, 0, bytes);
        g_hdf5.full_matrix_b = bytes;
    }

    h5_chunk_dimensions_calculate();

    if (!h5_file_setup())
    {
        return false;
    }

    g_hdf5.is_init = true;
    const char* file_type = g_hdf5.memory_map_required ? "Memory-mapped file" : "HDF5 file";
    print(VERBOSE, MSG_LOC(LAST), "%s has matrix size: %u x %u", file_type, mat_dim, mat_dim);

    return true;
}

#define H5_SEQUENCE_BATCH_SIZE 1 << 12

bool
h5_sequences_store(sequences_t sequences, sequence_count_t seq_count)
{
    if (!g_hdf5.mode_write || !g_hdf5.is_init || g_hdf5.sequences_stored)
    {
        return true;
    }

    if (g_hdf5.file_id < 0)
    {
        print(ERROR, MSG_NONE, "Cannot store sequences: HDF5 file not initialized");
        return false;
    }

    print(INFO, MSG_NONE, "Storing %u sequences in HDF5 file", seq_count);

    hid_t seq_group = H5Gcreate2(g_hdf5.file_id,
                                 "/sequences",
                                 H5P_DEFAULT,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);

    if (seq_group < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequences group");
        return false;
    }

    H5Gclose(seq_group);

    g_hdf5.seq_dims[0] = g_hdf5.matrix_dim;
    hid_t lengths_space = H5Screate_simple(1, g_hdf5.seq_dims, NULL);

    if (lengths_space < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequence lengths dataspace");
        return false;
    }

    g_hdf5.lengths_id = H5Dcreate2(g_hdf5.file_id,
                                   "/sequences/lengths",
                                   H5T_STD_U64LE,
                                   lengths_space,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT,
                                   H5P_DEFAULT);

    H5Sclose(lengths_space);

    if (g_hdf5.lengths_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequence lengths dataset");
        return false;
    }

    sequence_length_t* lengths = MALLOC(lengths, seq_count);
    if (!lengths)
    {
        print(ERROR, MSG_NONE, "Failed to allocate memory for sequence lengths");
        H5Dclose(g_hdf5.lengths_id);
        g_hdf5.lengths_id = H5I_INVALID_HID;
        return false;
    }

    for (sequence_index_t i = 0; i < seq_count; i++)
    {
        lengths[i] = sequences[i].length;
    }

    herr_t status = H5Dwrite(g_hdf5.lengths_id,
                             H5T_NATIVE_ULONG,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             lengths);

    free(lengths);

    if (status < 0)
    {
        print(ERROR, MSG_NONE, "Failed to write sequence lengths");
        H5Dclose(g_hdf5.lengths_id);
        g_hdf5.lengths_id = H5I_INVALID_HID;
        return false;
    }

    hid_t string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, H5T_VARIABLE);

    hid_t seq_space = H5Screate_simple(1, g_hdf5.seq_dims, NULL);
    if (seq_space < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequences dataspace");
        H5Tclose(string_type);
        H5Dclose(g_hdf5.lengths_id);
        g_hdf5.lengths_id = H5I_INVALID_HID;
        return false;
    }

    g_hdf5.sequences_id = H5Dcreate2(g_hdf5.file_id,
                                     "/sequences/dataset",
                                     string_type,
                                     seq_space,
                                     H5P_DEFAULT,
                                     H5P_DEFAULT,
                                     H5P_DEFAULT);

    if (g_hdf5.sequences_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequences dataset");
        H5Sclose(seq_space);
        H5Tclose(string_type);
        H5Dclose(g_hdf5.lengths_id);
        g_hdf5.lengths_id = H5I_INVALID_HID;
        return false;
    }

    const sequence_index_t batch_size = H5_SEQUENCE_BATCH_SIZE;
    for (sequence_index_t batch_start = 0; batch_start < seq_count; batch_start += batch_size)
    {
        sequence_index_t batch_end = MIN(batch_start + batch_size, seq_count);
        sequence_index_t current_batch_size = batch_end - batch_start;

        char** seq_data = MALLOC(seq_data, current_batch_size);
        if (!seq_data)
        {
            print(ERROR, MSG_NONE, "Failed to allocate memory for sequence batch");
            H5Sclose(seq_space);
            H5Tclose(string_type);
            H5Dclose(g_hdf5.sequences_id);
            g_hdf5.sequences_id = H5I_INVALID_HID;
            H5Dclose(g_hdf5.lengths_id);
            g_hdf5.lengths_id = H5I_INVALID_HID;
            return false;
        }

        for (sequence_index_t i = 0; i < current_batch_size; i++)
        {
            seq_data[i] = sequences[batch_start + i].letters;
        }

        hsize_t batch_dims[1] = { current_batch_size };
        hid_t batch_mem_space = H5Screate_simple(1, batch_dims, NULL);

        if (batch_mem_space < 0)
        {
            print(ERROR, MSG_NONE, "Failed to create memory dataspace for sequence batch");
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
        status = H5Sselect_hyperslab(seq_space, H5S_SELECT_SET, start, NULL, count, NULL);

        if (status >= 0)
        {
            status = H5Dwrite(g_hdf5.sequences_id,
                              string_type,
                              batch_mem_space,
                              seq_space,
                              H5P_DEFAULT,
                              seq_data);
        }

        H5Sclose(batch_mem_space);
        free(seq_data);

        if (status < 0)
        {
            print(ERROR, MSG_NONE, "Failed to write sequence batch");
            H5Sclose(seq_space);
            H5Tclose(string_type);
            H5Dclose(g_hdf5.sequences_id);
            g_hdf5.sequences_id = H5I_INVALID_HID;
            H5Dclose(g_hdf5.lengths_id);
            g_hdf5.lengths_id = H5I_INVALID_HID;
            return false;
        }

        const int percentage = (int)(100 * batch_end / seq_count);
        print(PROGRESS, MSG_PERCENT(percentage), "Storing sequences");
    }

    H5Sclose(seq_space);
    H5Tclose(string_type);
    g_hdf5.sequences_stored = true;
    return true;
}

void
h5_matrix_set(sequence_index_t row, sequence_index_t col, score_t value)
{
    if (!g_hdf5.mode_write)
    {
        return;
    }

    if (g_hdf5.memory_map_required)
    {
        g_hdf5.memory_map.matrix[matrix_triangle_index(row, col)] = value;
    }

    else
    {
        g_hdf5.full_matrix[row * g_hdf5.matrix_dim + col] = value;
        g_hdf5.full_matrix[col * g_hdf5.matrix_dim + row] = value;
    }
}

void
h5_checksum_set(int64_t checksum)
{
    g_hdf5.checksum = checksum;
}

int64_t
h5_checksum(void)
{
    return g_hdf5.checksum;
}

static void h5_store_checksum(void);
static void h5_flush_memory_map(void);
static void h5_flush_full_matrix(void);
static void h5_file_close(void);

void
h5_close(int skip_flush)
{
    if (!g_hdf5.is_init)
    {
        return;
    }

    if (g_hdf5.mode_write)
    {
        if (!skip_flush)
        {
            print(SECTION, MSG_NONE, "Finalizing Results");
            print(INFO, MSG_LOC(FIRST), "Matrix checksum: %lld", g_hdf5.checksum);
            print(INFO, MSG_LOC(LAST), "Writing results to %s", file_name_path(g_hdf5.file_path));
            print_error_prefix("HDF5");
            h5_store_checksum();
            g_hdf5.memory_map_required ? h5_flush_memory_map() : h5_flush_full_matrix();
        }

        h5_file_close();
    }

    g_hdf5.is_init = false;
}

#ifdef USE_CUDA

score_t*
h5_matrix_data(void)
{
    if (!g_hdf5.mode_write)
    {
        return NULL;
    }

    return g_hdf5.memory_map_required ? g_hdf5.memory_map.matrix : g_hdf5.full_matrix;
}

size_t
h5_matrix_bytes(void)
{
    if (!g_hdf5.mode_write)
    {
        return 0;
    }

    return g_hdf5.memory_map_required ? g_hdf5.memory_map.meta.bytes : g_hdf5.full_matrix_b;
}

bool
h5_triangle_indices_64_bit(void)
{
    return g_hdf5.matrix_dim >= 92683; // (matrix_dim - 1) * (matrix_dim - 2) / 2 > UINT32_MAX
}

static bool h5_triangle_indices_calculate(void);

half_t*
h5_triangle_indices_32(void)
{
    if (!g_hdf5.mode_write || h5_triangle_indices_64_bit())
    {
        return NULL;
    }

    if (!g_hdf5.triangle_indices_32 && !h5_triangle_indices_calculate())
    {
        print(ERROR, MSG_NONE, "Failed to calculate result offsets");
        return NULL;
    }

    return g_hdf5.triangle_indices_32;
}

size_t*
h5_triangle_indices_64(void)
{
    if (!g_hdf5.mode_write || !h5_triangle_indices_64_bit())
    {
        return NULL;
    }

    if (!g_hdf5.triangle_indices_64 && !h5_triangle_indices_calculate())
    {
        print(ERROR, MSG_NONE, "Failed to calculate result offsets");
        return NULL;
    }

    return g_hdf5.triangle_indices_64;
}

static bool
h5_triangle_indices_calculate(void)
{
    if (!g_hdf5.mode_write || !g_hdf5.is_init)
    {
        return false;
    }

    if (g_hdf5.triangle_indices_64 || g_hdf5.triangle_indices_32)
    {
        return true;
    }

    sequence_count_t sequence_count = g_hdf5.matrix_dim;

    if (h5_triangle_indices_64_bit())
    {
        if (!(g_hdf5.triangle_indices_64 = MALLOC(g_hdf5.triangle_indices_64, sequence_count)))
        {
            print(ERROR, MSG_NONE, "Failed to allocate memory for result offsets");
            return false;
        }

        for (sequence_index_t i = 0; i < sequence_count; i++)
        {
            g_hdf5.triangle_indices_64[i] = (i * (i - 1)) / 2;
        }
    }

    else
    {
        if (!(g_hdf5.triangle_indices_32 = MALLOC(g_hdf5.triangle_indices_32, sequence_count)))
        {
            print(ERROR, MSG_NONE, "Failed to allocate memory for result offsets");
            return false;
        }

        for (sequence_index_t i = 0; i < sequence_count; i++)
        {
            g_hdf5.triangle_indices_32[i] = (i * (i - 1)) / 2;
        }
    }

    return true;
}

#endif

#define H5_MIN_CHUNK_SIZE 1 << 7
#define H5_MAX_CHUNK_SIZE H5_MIN_CHUNK_SIZE << 7

static void
h5_chunk_dimensions_calculate(void)
{
    sequence_count_t matrix_dim = g_hdf5.matrix_dim;
    sequence_count_t chunk_dim;

    if (matrix_dim <= H5_MIN_CHUNK_SIZE)
    {
        chunk_dim = matrix_dim;
    }

    else if (matrix_dim > H5_MAX_CHUNK_SIZE)
    {
        sequence_count_t target_chunks = matrix_dim > 1 << 15 ? 1 << 4 : 1 << 5;
        chunk_dim = matrix_dim / target_chunks;
        chunk_dim = ALIGN_POW2(chunk_dim, H5_MIN_CHUNK_SIZE);
        chunk_dim = MAX(chunk_dim, H5_MIN_CHUNK_SIZE);
        chunk_dim = MIN(chunk_dim, H5_MAX_CHUNK_SIZE);
    }

    else
    {
        sequence_count_t chunk_candidates[] = { H5_MIN_CHUNK_SIZE,      H5_MIN_CHUNK_SIZE << 1,
                                                H5_MIN_CHUNK_SIZE << 2, H5_MIN_CHUNK_SIZE << 3,
                                                H5_MIN_CHUNK_SIZE << 4, H5_MIN_CHUNK_SIZE << 5,
                                                H5_MIN_CHUNK_SIZE << 6, H5_MAX_CHUNK_SIZE };

        sequence_count_t num_candidates = sizeof(chunk_candidates) / sizeof(*chunk_candidates);

        chunk_dim = H5_MIN_CHUNK_SIZE;

        for (sequence_index_t i = 0; i < num_candidates; i++)
        {
            sequence_count_t candidate = chunk_candidates[i];
            if (candidate > H5_MAX_CHUNK_SIZE || candidate > matrix_dim)
            {
                break;
            }

            chunk_dim = candidate;
            if (candidate * 8 >= matrix_dim)
            {
                break;
            }
        }
    }

    g_hdf5.chunk_dims[0] = chunk_dim;
    g_hdf5.chunk_dims[1] = chunk_dim;
    print(VERBOSE, MSG_LOC(FIRST), "HDF5 chunk size: %u x %u", chunk_dim, chunk_dim);
}

static bool
h5_file_setup(void)
{
    if (!g_hdf5.mode_write)
    {
        return true;
    }

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);

    g_hdf5.file_id = H5Fcreate(g_hdf5.file_path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    if (g_hdf5.file_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create HDF5 file: %s", g_hdf5.file_path);
        h5_file_close();
        return false;
    }

    g_hdf5.matrix_dims[0] = g_hdf5.matrix_dim;
    g_hdf5.matrix_dims[1] = g_hdf5.matrix_dim;
    hid_t matrix_space = H5Screate_simple(2, g_hdf5.matrix_dims, NULL);

    if (matrix_space < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create matrix dataspace");
        h5_file_close();
        return false;
    }

    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);

    if (g_hdf5.matrix_dim > H5_MIN_CHUNK_SIZE)
    {
        H5Pset_chunk(plist_id, 2, g_hdf5.chunk_dims);

        if (g_hdf5.compression_level > 0)
        {
            H5Pset_deflate(plist_id, g_hdf5.compression_level);
        }
    }

    g_hdf5.matrix_id = H5Dcreate2(g_hdf5.file_id,
                                  "/similarity_matrix",
                                  H5T_STD_I32LE,
                                  matrix_space,
                                  H5P_DEFAULT,
                                  plist_id,
                                  H5P_DEFAULT);

    H5Sclose(matrix_space);
    H5Pclose(plist_id);

    if (g_hdf5.matrix_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create similarity matrix dataset");
        h5_file_close();
        return false;
    }

    return true;
}

static void
h5_file_close(void)
{
    if (g_hdf5.memory_map_required)
    {
        file_matrix_close(&g_hdf5.memory_map);
        remove(g_hdf5.file_matrix_name);
    }

    else
    {
        if (g_hdf5.full_matrix)
        {
            aligned_free(g_hdf5.full_matrix);
            g_hdf5.full_matrix = NULL;
        }

        g_hdf5.full_matrix_b = 0;
    }

#ifdef USE_CUDA
    if (g_hdf5.triangle_indices_64)
    {
        free(g_hdf5.triangle_indices_64);
        g_hdf5.triangle_indices_64 = NULL;
    }

    if (g_hdf5.triangle_indices_32)
    {
        free(g_hdf5.triangle_indices_32);
        g_hdf5.triangle_indices_32 = NULL;
    }

#endif

    if (g_hdf5.sequences_id > 0)
    {
        H5Dclose(g_hdf5.sequences_id);
        g_hdf5.sequences_id = H5I_INVALID_HID;
    }

    if (g_hdf5.lengths_id > 0)
    {
        H5Dclose(g_hdf5.lengths_id);
        g_hdf5.lengths_id = H5I_INVALID_HID;
    }

    if (g_hdf5.matrix_id > 0)
    {
        H5Dclose(g_hdf5.matrix_id);
        g_hdf5.matrix_id = H5I_INVALID_HID;
    }

    if (g_hdf5.file_id > 0)
    {
        H5Fclose(g_hdf5.file_id);
        g_hdf5.file_id = H5I_INVALID_HID;
    }
}

static void
h5_store_checksum(void)
{
    if (!g_hdf5.mode_write || g_hdf5.matrix_id < 0 || g_hdf5.file_id < 0)
    {
        return;
    }

    htri_t attr_exists = H5Aexists(g_hdf5.matrix_id, "checksum");
    if (attr_exists > 0)
    {
        H5Adelete(g_hdf5.matrix_id, "checksum");
    }

    hid_t attr_space = H5Screate(H5S_SCALAR);
    if (attr_space < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create dataspace for checksum attribute");
        return;
    }

    hid_t attr_id = H5Acreate2(g_hdf5.matrix_id,
                               "checksum",
                               H5T_STD_I64LE,
                               attr_space,
                               H5P_DEFAULT,
                               H5P_DEFAULT);

    if (attr_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create checksum attribute");
        H5Sclose(attr_space);
        return;
    }

    herr_t status = H5Awrite(attr_id, H5T_NATIVE_INT64, &g_hdf5.checksum);
    H5Aclose(attr_id);
    H5Sclose(attr_space);

    if (status < 0)
    {
        print(ERROR, MSG_NONE, "Failed to write checksum attribute");
        return;
    }

    return;
}

static void
h5_flush_full_matrix(void)
{
    herr_t status = H5Dwrite(g_hdf5.matrix_id,
                             H5T_NATIVE_INT,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             g_hdf5.full_matrix);

    if (status < 0)
    {
        print(ERROR, MSG_NONE, "Failed to write matrix data to HDF5");
        return;
    }

    return;
}

static void
h5_flush_memory_map(void)
{
    sequence_count_t matrix_dim = g_hdf5.matrix_dim;
    size_t available_mem = available_memory();
    if (!available_mem)
    {
        print(ERROR, MSG_NONE, "Failed to retrieve available memory");
        return;
    }

    hsize_t chunk_rows = g_hdf5.chunk_dims[0];
    size_t row_bytes = matrix_dim * sizeof(*g_hdf5.memory_map.matrix);
    sequence_count_t max_rows = (sequence_count_t)(available_mem / (4 * row_bytes));
    sequence_count_t chunk_size = (sequence_count_t)(chunk_rows > 4 ? chunk_rows : 4);
    if (chunk_size > max_rows && max_rows > 4)
    {
        chunk_size = max_rows;
    }

    const size_t buffer_mib = (chunk_size * row_bytes) / MiB;
    print(VERBOSE, MSG_NONE, "Using %u rows per chunk (%zu MiB buffer)", chunk_size, buffer_mib);

    score_t* buffer = CAST(buffer)(calloc(chunk_size, row_bytes));
    if (!buffer)
    {
        print(WARNING, MSG_NONE, "Failed to allocate buffer of %zu bytes", row_bytes * chunk_size);

        chunk_size = 1;
        buffer = CAST(buffer)(calloc(chunk_size, row_bytes));
        if (!buffer)
        {
            print(ERROR, MSG_NONE, "Cannot allocate even minimal buffer, aborting");
            return;
        }

        print(WARNING, MSG_NONE, "Using minimal buffer size of 1 row (%zu bytes)", row_bytes);
    }

    hid_t file_space = H5Dget_space(g_hdf5.matrix_id);
    if (file_space < 0)
    {
        free(buffer);
        return;
    }

    print(INFO, MSG_NONE, "Converting memory-mapped matrix to HDF5 format");

    for (sequence_index_t begin = 0; begin < matrix_dim; begin += chunk_size)
    {
        sequence_count_t end = MIN(begin + chunk_size, matrix_dim);

        for (sequence_index_t i = begin; i < end; i++)
        {
            alignment_size_t row_offset = (alignment_size_t)(i - begin) * matrix_dim;

            for (sequence_index_t j = i + 1; j < matrix_dim; j++)
            {
                buffer[row_offset + j] = g_hdf5.memory_map.matrix[matrix_triangle_index(i, j)];
            }

            for (sequence_index_t j = 0; j < i; j++)
            {
                if (j >= begin)
                {
                    buffer[row_offset + j] = buffer[(j - begin) * matrix_dim + i];
                }

                else
                {
                    buffer[row_offset + j] = g_hdf5.memory_map.matrix[matrix_triangle_index(j, i)];
                }
            }
        }

        sequence_count_t rows = end - begin;
        hsize_t start[2] = { begin, 0 };
        hsize_t count[2] = { rows, matrix_dim };
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);

        hsize_t mem_dims[2] = { rows, matrix_dim };
        hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);

        if (mem_space < 0)
        {
            print(ERROR, MSG_NONE, "Failed to create memory dataspace for matrix chunk");
            H5Sclose(file_space);
            free(buffer);
            return;
        }

        herr_t status = H5Dwrite(g_hdf5.matrix_id,
                                 H5T_NATIVE_INT,
                                 mem_space,
                                 file_space,
                                 H5P_DEFAULT,
                                 buffer);

        H5Sclose(mem_space);

        if (status < 0)
        {
            print(ERROR, MSG_NONE, "Failed to write chunk to HDF5");
            H5Sclose(file_space);
            free(buffer);
            return;
        }

        const int percentage = (int)(100 * end / matrix_dim);
        print(PROGRESS, MSG_PERCENT(percentage), "Converting to HDF5");
    }

    H5Sclose(file_space);
    free(buffer);
    return;
}
