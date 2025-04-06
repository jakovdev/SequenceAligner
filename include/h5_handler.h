#ifndef H5_HANDLER_H
#define H5_HANDLER_H

#include "files.h"
#include "sequence.h"
#include <hdf5.h>
#include <math.h>

#define H5_MIN_CHUNK_SIZE 128
#define H5_MAX_CHUNK_SIZE 1024
#define H5_SEQUENCE_BATCH_SIZE 5000
#define MMAP_MEMORY_USAGE_THRESHOLD 0.7

typedef struct
{
    int* data;
    size_t size;
} MatrixBuffer;

typedef struct
{
    hid_t file_id;
    hid_t matrix_dataset_id;
    hid_t seq_dataset_id;
    hid_t seq_lengths_dataset_id;

    hsize_t matrix_dims[2];
    hsize_t chunk_dims[2];
    hsize_t seq_dims[1];

    size_t matrix_size;

    MatrixBuffer buffer;
    MmapMatrix mmap_matrix;
    bool use_mmap;
    char mmap_filename[MAX_PATH];

    int64_t checksum;

    bool sequences_stored;
    bool is_init;
} H5Handler;

INLINE bool
matrix_buffer_allocate(MatrixBuffer* buffer, size_t matrix_size)
{
    size_t bytes = matrix_size * matrix_size * sizeof(int);
    buffer->data = alloc_huge_page(bytes);
    if (!buffer->data)
    {
        return false;
    }

    memset(buffer->data, 0, bytes);
    buffer->size = matrix_size;

    return true;
}

INLINE void
matrix_buffer_free(MatrixBuffer* buffer)
{
    if (buffer->data)
    {
        aligned_free(buffer->data);
        buffer->data = NULL;
    }

    buffer->size = 0;
}

INLINE void
h5_calculate_chunk_dimensions(H5Handler* handler)
{
    size_t matrix_size = handler->matrix_size;
    size_t optimal_chunk;

    if (matrix_size <= 1000)
    {
        optimal_chunk = matrix_size;
    }

    else
    {
        double scale_factor = 1.0 - fmin(0.9, log10((double)matrix_size / 1000) * 0.3);
        optimal_chunk = (size_t)(1000 * scale_factor);
    }

    optimal_chunk = max(H5_MIN_CHUNK_SIZE, min(optimal_chunk, H5_MAX_CHUNK_SIZE));

    handler->chunk_dims[0] = optimal_chunk;
    handler->chunk_dims[1] = optimal_chunk;

    print(VERBOSE,
          MSG_LOC(FIRST),
          "HDF5 chunk size: %zu x %zu for %zu x %zu matrix",
          optimal_chunk,
          optimal_chunk,
          matrix_size,
          matrix_size);
}

INLINE bool
h5_create_file(H5Handler* handler)
{
    if (!args_mode_write())
    {
        return true;
    }

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);

    handler->file_id = H5Fcreate(args_path_output(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    if (handler->file_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create HDF5 file: %s", args_path_output());
        return false;
    }

    return true;
}

INLINE bool
h5_create_matrix_dataset(H5Handler* handler)
{
    handler->matrix_dims[0] = handler->matrix_size;
    handler->matrix_dims[1] = handler->matrix_size;
    hid_t matrix_space = H5Screate_simple(2, handler->matrix_dims, NULL);

    if (matrix_space < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create matrix dataspace");
        H5Fclose(handler->file_id);
        handler->file_id = -1;
        return false;
    }

    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, 2, handler->chunk_dims);

    int compression_level = args_compression_level();
    if (compression_level > 0)
    {
        H5Pset_deflate(plist_id, compression_level);
    }

    handler->matrix_dataset_id = H5Dcreate2(handler->file_id,
                                            "/similarity_matrix",
                                            H5T_STD_I32LE,
                                            matrix_space,
                                            H5P_DEFAULT,
                                            plist_id,
                                            H5P_DEFAULT);

    H5Sclose(matrix_space);

    if (handler->matrix_dataset_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create similarity matrix dataset");
        H5Pclose(plist_id);
        H5Fclose(handler->file_id);
        handler->file_id = -1;
        return false;
    }

    H5Pclose(plist_id);
    return true;
}

INLINE bool
h5_create_sequence_group(H5Handler* handler)
{
    hid_t seq_group = H5Gcreate2(handler->file_id,
                                 "/sequences",
                                 H5P_DEFAULT,
                                 H5P_DEFAULT,
                                 H5P_DEFAULT);

    if (seq_group < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequences group");
        H5Dclose(handler->matrix_dataset_id);
        handler->matrix_dataset_id = -1;
        H5Fclose(handler->file_id);
        handler->file_id = -1;
        return false;
    }

    H5Gclose(seq_group);
    return true;
}

INLINE bool
h5_create_sequence_length_dataset(H5Handler* handler)
{
    handler->seq_dims[0] = handler->matrix_size;
    hid_t seq_lengths_space = H5Screate_simple(1, handler->seq_dims, NULL);

    if (seq_lengths_space < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequence lengths dataspace");
        H5Dclose(handler->matrix_dataset_id);
        handler->matrix_dataset_id = -1;
        H5Fclose(handler->file_id);
        handler->file_id = -1;
        return false;
    }

    handler->seq_lengths_dataset_id = H5Dcreate2(handler->file_id,
                                                 "/sequences/lengths",
                                                 H5T_STD_U64LE,
                                                 seq_lengths_space,
                                                 H5P_DEFAULT,
                                                 H5P_DEFAULT,
                                                 H5P_DEFAULT);

    H5Sclose(seq_lengths_space);

    if (handler->seq_lengths_dataset_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequence lengths dataset");
        H5Dclose(handler->matrix_dataset_id);
        handler->matrix_dataset_id = -1;
        H5Fclose(handler->file_id);
        handler->file_id = -1;
        return false;
    }

    return true;
}

INLINE bool
h5_setup_file(H5Handler* handler)
{
    if (!args_mode_write())
    {
        return true;
    }

    if (!h5_create_file(handler))
    {
        return false;
    }

    if (!h5_create_matrix_dataset(handler))
    {
        return false;
    }

    if (!h5_create_sequence_group(handler))
    {
        return false;
    }

    if (!h5_create_sequence_length_dataset(handler))
    {
        return false;
    }

    return true;
}

INLINE bool
h5_initialize_memory(H5Handler* handler)
{
    if (handler->use_mmap)
    {
        mmap_matrix_file_name(handler->mmap_filename, MAX_PATH, args_path_output());

        print(INFO,
              MSG_LOC(FIRST),
              "Matrix size exceeds memory threshold, using memory-mapped file");

        handler->mmap_matrix = mmap_matrix_create(handler->mmap_filename, handler->matrix_size);
        return handler->mmap_matrix.data != NULL;
    }

    else
    {
        return matrix_buffer_allocate(&handler->buffer, handler->matrix_size);
    }
}

INLINE void
h5_cleanup_on_init_failure(H5Handler* handler)
{
    if (handler->use_mmap)
    {
        mmap_matrix_close(&handler->mmap_matrix);
        remove(handler->mmap_filename);
    }

    else
    {
        matrix_buffer_free(&handler->buffer);
    }

    if (handler->matrix_dataset_id > 0)
    {
        H5Dclose(handler->matrix_dataset_id);
    }

    if (handler->seq_lengths_dataset_id > 0)
    {
        H5Dclose(handler->seq_lengths_dataset_id);
    }

    if (handler->file_id > 0)
    {
        H5Fclose(handler->file_id);
    }
}

INLINE void
h5_initialize(H5Handler* handler, size_t matrix_size)
{
    handler->matrix_size = matrix_size;
    handler->file_id = -1;
    handler->matrix_dataset_id = -1;
    handler->seq_dataset_id = -1;
    handler->seq_lengths_dataset_id = -1;

    if (!args_mode_write())
    {
        handler->is_init = true;
        return;
    }

    const size_t bytes_needed = matrix_size * matrix_size * sizeof(int);
    const size_t safe_memory = available_memory() * MMAP_MEMORY_USAGE_THRESHOLD;

    handler->use_mmap = bytes_needed > safe_memory;

    if (!h5_initialize_memory(handler))
    {
        print(ERROR, MSG_NONE, "Failed to initialize matrix memory");
        return;
    }

    h5_calculate_chunk_dimensions(handler);

    if (!h5_setup_file(handler))
    {
        h5_cleanup_on_init_failure(handler);
        return;
    }

    handler->is_init = true;

    print(VERBOSE,
          MSG_LOC(LAST),
          "%s file created with matrix size: %zu x %zu",
          handler->use_mmap ? "Memory-mapped" : "HDF5",
          matrix_size,
          matrix_size);

    return;
}

INLINE bool
h5_flush_mmap_to_hdf5(H5Handler* handler)
{
    size_t matrix_size = handler->matrix_size;
    size_t available_mem = available_memory();
    if (!available_mem)
    {
        print(ERROR, MSG_NONE, "Failed to retrieve available memory");
        return false;
    }

    size_t row_bytes = matrix_size * sizeof(int);
    size_t max_rows = available_mem / (4 * row_bytes);
    size_t chunk_size = max_rows < 4 ? 4 : max_rows;
    chunk_size = chunk_size > KiB ? KiB : chunk_size;

    print(VERBOSE,
          MSG_NONE,
          "Converting matrix using %zu rows per chunk (%zu MiB buffer)",
          chunk_size,
          (chunk_size * row_bytes) / MiB);

    int* buffer = calloc(chunk_size, row_bytes);
    if (!buffer)
    {
        print(WARNING,
              MSG_NONE,
              "Failed to allocate transfer buffer of %zu bytes",
              chunk_size * row_bytes);

        chunk_size = 1;
        buffer = calloc(chunk_size, row_bytes);
        if (!buffer)
        {
            print(ERROR, MSG_NONE, "Cannot allocate even minimal buffer, aborting");
            return false;
        }

        print(WARNING, MSG_NONE, "Using minimal buffer size of 1 row (%zu bytes)", row_bytes);
    }

    hid_t file_space = H5Dget_space(handler->matrix_dataset_id);
    if (file_space < 0)
    {
        free(buffer);
        return false;
    }

    print(INFO, MSG_NONE, "Converting memory-mapped matrix to HDF5 format");

    for (size_t start_row = 0; start_row < matrix_size; start_row += chunk_size)
    {
        size_t end_row = start_row + chunk_size;
        if (end_row > matrix_size)
        {
            end_row = matrix_size;
        }

        size_t rows = end_row - start_row;

        for (size_t i = start_row; i < end_row; i++)
        {
            for (size_t j = i; j < matrix_size; j++)
            {
                size_t idx = mmap_triangle_index(i, j, matrix_size);
                int value = handler->mmap_matrix.data[idx];
                buffer[(i - start_row) * matrix_size + j] = value;
            }

            for (size_t j = 0; j < i; j++)
            {
                if (j >= start_row)
                {
                    buffer[(i - start_row) * matrix_size +
                           j] = buffer[(j - start_row) * matrix_size + i];
                }

                else
                {
                    size_t idx = mmap_triangle_index(j, i, matrix_size);
                    int value = handler->mmap_matrix.data[idx];
                    buffer[(i - start_row) * matrix_size + j] = value;
                }
            }
        }

        hsize_t start[2] = { start_row, 0 };
        hsize_t count[2] = { rows, matrix_size };
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);

        hsize_t mem_dims[2] = { rows, matrix_size };
        hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);

        if (mem_space < 0)
        {
            print(ERROR, MSG_NONE, "Failed to create memory dataspace for matrix chunk");
            H5Sclose(file_space);
            free(buffer);
            return false;
        }

        herr_t status = H5Dwrite(handler->matrix_dataset_id,
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
            return false;
        }

        print(PROGRESS, MSG_PROPORTION((float)end_row / matrix_size), "Converting to HDF5");
    }

    H5Sclose(file_space);
    free(buffer);
    return true;
}

INLINE bool
h5_flush_buffer_to_hdf5(H5Handler* handler)
{
    herr_t status = H5Dwrite(handler->matrix_dataset_id,
                             H5T_NATIVE_INT,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             handler->buffer.data);

    if (status < 0)
    {
        print(ERROR, MSG_NONE, "Failed to write matrix data to HDF5");
        return false;
    }

    return true;
}

INLINE bool
h5_flush_matrix(H5Handler* handler)
{
    if (!args_mode_write() || !handler->is_init)
    {
        return true;
    }

    if (handler->matrix_dataset_id < 0 || (handler->use_mmap && !handler->mmap_matrix.data) ||
        (!handler->use_mmap && !handler->buffer.data))
    {
        print(ERROR, MSG_NONE, "Cannot flush matrix: HDF5 resources not properly initialized");
        return false;
    }

    bool result;

    if (handler->use_mmap)
    {
        result = h5_flush_mmap_to_hdf5(handler);
    }

    else
    {
        result = h5_flush_buffer_to_hdf5(handler);
    }

    return result;
}

INLINE void
h5_set_matrix_value(H5Handler* handler, size_t row, size_t col, int value)
{
    if (!args_mode_write() || !handler->is_init || row >= handler->matrix_size ||
        col >= handler->matrix_size)
    {
        return;
    }

    if (handler->use_mmap)
    {
        if (row > col)
        {
            return;
        }

        if (!handler->mmap_matrix.data)
        {
            return;
        }

        mmap_matrix_set_value(&handler->mmap_matrix, row, col, value);
    }

    else
    {
        if (!handler->buffer.data)
        {
            return;
        }

        size_t pos1 = row * handler->matrix_size + col;
        handler->buffer.data[pos1] = value;

        if (row != col)
        {
            size_t pos2 = col * handler->matrix_size + row;
            handler->buffer.data[pos2] = value;
        }
    }
}

INLINE bool
h5_store_sequence_lengths(H5Handler* handler, Sequence* sequences, size_t seq_count)
{
    size_t* lengths = malloc(seq_count * sizeof(*lengths));
    if (!lengths)
    {
        print(ERROR, MSG_NONE, "Failed to allocate memory for sequence lengths");
        return false;
    }

    for (size_t i = 0; i < seq_count; i++)
    {
        lengths[i] = sequences[i].length;
    }

    herr_t status = H5Dwrite(handler->seq_lengths_dataset_id,
                             H5T_NATIVE_ULONG,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             lengths);

    free(lengths);

    if (status < 0)
    {
        print(ERROR, MSG_NONE, "Failed to write sequence lengths");
        return false;
    }

    return true;
}

INLINE bool
h5_store_sequence_batch(H5Handler* handler,
                        Sequence* sequences,
                        size_t batch_start,
                        size_t batch_end,
                        hid_t string_type,
                        hid_t seq_space)
{
    size_t current_batch_size = batch_end - batch_start;

    char** seq_data = malloc(current_batch_size * sizeof(*seq_data));
    if (!seq_data)
    {
        print(ERROR, MSG_NONE, "Failed to allocate memory for sequence batch");
        return false;
    }

    for (size_t i = 0; i < current_batch_size; i++)
    {
        seq_data[i] = sequences[batch_start + i].data;
    }

    hsize_t batch_dims[1] = { current_batch_size };
    hid_t batch_mem_space = H5Screate_simple(1, batch_dims, NULL);

    if (batch_mem_space < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create memory dataspace for sequence batch");
        free(seq_data);
        return false;
    }

    hsize_t start[1] = { batch_start };
    hsize_t count[1] = { current_batch_size };
    herr_t status = H5Sselect_hyperslab(seq_space, H5S_SELECT_SET, start, NULL, count, NULL);

    if (status < 0)
    {
        print(ERROR, MSG_NONE, "Failed to select hyperslab for sequence batch");
        H5Sclose(batch_mem_space);
        free(seq_data);
        return false;
    }

    status = H5Dwrite(handler->seq_dataset_id,
                      string_type,
                      batch_mem_space,
                      seq_space,
                      H5P_DEFAULT,
                      seq_data);

    H5Sclose(batch_mem_space);
    free(seq_data);

    if (status < 0)
    {
        print(ERROR, MSG_NONE, "Failed to write sequence batch");
        return false;
    }

    return true;
}

INLINE hid_t
h5_create_sequence_dataset(H5Handler* handler, hid_t string_type)
{
    hid_t seq_space = H5Screate_simple(1, handler->seq_dims, NULL);

    if (seq_space < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequence dataspace");
        H5Tclose(string_type);
        return -1;
    }

    handler->seq_dataset_id = H5Dcreate2(handler->file_id,
                                         "/sequences/data",
                                         string_type,
                                         seq_space,
                                         H5P_DEFAULT,
                                         H5P_DEFAULT,
                                         H5P_DEFAULT);

    if (handler->seq_dataset_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create sequences dataset");
        H5Sclose(seq_space);
        H5Tclose(string_type);
        return -1;
    }

    return seq_space;
}

INLINE bool
h5_store_sequences(H5Handler* handler, Sequence* sequences, size_t seq_count)
{
    if (!args_mode_write() || !handler->is_init || handler->sequences_stored)
    {
        return true;
    }

    if (handler->file_id < 0)
    {
        print(ERROR, MSG_NONE, "Cannot store sequences: HDF5 file not initialized");
        return false;
    }

    print(INFO, MSG_NONE, "Storing %zu sequences in HDF5 file", seq_count);

    hid_t string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, H5T_VARIABLE);

    if (!h5_store_sequence_lengths(handler, sequences, seq_count))
    {
        H5Tclose(string_type);
        return false;
    }

    hid_t seq_space = h5_create_sequence_dataset(handler, string_type);
    if (seq_space < 0)
    {
        return false;
    }

    const size_t batch_size = H5_SEQUENCE_BATCH_SIZE;

    for (size_t batch_start = 0; batch_start < seq_count; batch_start += batch_size)
    {
        size_t batch_end = batch_start + batch_size;
        if (batch_end > seq_count)
        {
            batch_end = seq_count;
        }

        if (!h5_store_sequence_batch(handler,
                                     sequences,
                                     batch_start,
                                     batch_end,
                                     string_type,
                                     seq_space))
        {
            H5Sclose(seq_space);
            H5Tclose(string_type);
            H5Dclose(handler->seq_dataset_id);
            handler->seq_dataset_id = -1;
            return false;
        }

        print(PROGRESS, MSG_PROPORTION((float)batch_end / seq_count), "Storing sequences");
    }

    H5Sclose(seq_space);
    H5Tclose(string_type);

    handler->sequences_stored = true;

    return true;
}

INLINE bool
h5_store_checksum(H5Handler* handler)
{
    if (!args_mode_write() || handler->matrix_dataset_id < 0 || handler->file_id < 0)
    {
        return false;
    }

    htri_t attr_exists = H5Aexists(handler->matrix_dataset_id, "checksum");
    if (attr_exists > 0)
    {
        H5Adelete(handler->matrix_dataset_id, "checksum");
    }

    hid_t attr_space = H5Screate(H5S_SCALAR);
    if (attr_space < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create dataspace for checksum attribute");
        return false;
    }

    hid_t attr_id = H5Acreate2(handler->matrix_dataset_id,
                               "checksum",
                               H5T_STD_I64LE,
                               attr_space,
                               H5P_DEFAULT,
                               H5P_DEFAULT);

    if (attr_id < 0)
    {
        print(ERROR, MSG_NONE, "Failed to create checksum attribute");
        H5Sclose(attr_space);
        return false;
    }

    herr_t status = H5Awrite(attr_id, H5T_NATIVE_INT64, &handler->checksum);

    H5Aclose(attr_id);
    H5Sclose(attr_space);

    if (status < 0)
    {
        print(ERROR, MSG_NONE, "Failed to write checksum attribute");
        return false;
    }

    return true;
}

INLINE void
h5_close(H5Handler* handler)
{
    if (!handler->is_init)
    {
        return;
    }

    if (args_mode_write())
    {
        bool success = true;

        print(SECTION, MSG_NONE, "Finalizing Results");
        print(INFO,
              MSG_LOC(FIRST),
              "Writing results to output file: %s",
              file_name_path(args_path_output()));

        if (!h5_flush_matrix(handler))
        {
            print(ERROR, MSG_NONE, "Failed to write matrix data to output file");
            success = false;
        }

        print(INFO, MSG_LOC(LAST), "Matrix checksum: %lld", handler->checksum);
        if (!h5_store_checksum(handler))
        {
            print(ERROR, MSG_NONE, "Failed to store checksum in output file");
            success = false;
        }

        if (handler->seq_dataset_id > 0)
        {
            H5Dclose(handler->seq_dataset_id);
            handler->seq_dataset_id = -1;
        }

        if (handler->seq_lengths_dataset_id > 0)
        {
            H5Dclose(handler->seq_lengths_dataset_id);
            handler->seq_lengths_dataset_id = -1;
        }

        if (handler->matrix_dataset_id > 0)
        {
            H5Dclose(handler->matrix_dataset_id);
            handler->matrix_dataset_id = -1;
        }

        if (handler->file_id > 0)
        {
            H5Fclose(handler->file_id);
            handler->file_id = -1;
        }

        if (handler->use_mmap)
        {
            mmap_matrix_close(&handler->mmap_matrix);
            if (success)
            {
                if (remove(handler->mmap_filename) != 0)
                {
                    print(WARNING,
                          MSG_NONE,
                          "Failed to remove temporary file: %s",
                          handler->mmap_filename);
                }
            }

            else
            {
                print(WARNING,
                      MSG_NONE,
                      "Keeping temporary file for debugging: %s",
                      handler->mmap_filename);
            }
        }

        else
        {
            matrix_buffer_free(&handler->buffer);
        }
    }

    handler->is_init = false;
}

#endif // H5_HANDLER_H