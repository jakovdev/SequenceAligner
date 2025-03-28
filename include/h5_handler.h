#ifndef H5_HANDLER_H
#define H5_HANDLER_H

#include "common.h"
#include "benchmark.h"
#include "sequence.h"
#include <hdf5.h>

#define H5_MIN_CHUNK_SIZE 128
#define H5_MAX_CHUNK_SIZE 1024
#define H5_SEQUENCE_BATCH_SIZE 5000

typedef struct {
    int* data;
    size_t size;
} MatrixBuffer;

typedef struct {
    hid_t file_id;
    hid_t matrix_dataset_id;
    hid_t seq_dataset_id;
    hid_t seq_lengths_dataset_id;
    
    hsize_t matrix_dims[2];
    hsize_t chunk_dims[2];
    hsize_t seq_dims[1];
    
    size_t matrix_size;
    size_t max_seq_length;

    MatrixBuffer buffer;
    int64_t matrix_checksum;
    
    bool sequences_stored;
    bool is_init;
} H5Handler;

INLINE bool init_matrix_buffer(MatrixBuffer* buffer, size_t matrix_size) {
    size_t bytes = matrix_size * matrix_size * sizeof(int);
    buffer->data = (int*)huge_page_alloc(bytes);
    if (!buffer->data) return false;
    
    memset(buffer->data, 0, bytes);
    
    buffer->size = matrix_size;
    
    return true;
}

INLINE void free_matrix_buffer(MatrixBuffer* buffer) {
    if (buffer->data) {
        aligned_free(buffer->data);
        buffer->data = NULL;
    }
    
    buffer->size = 0;
}

INLINE void calculate_chunk_dimensions(H5Handler* handler) {
    size_t matrix_size = handler->matrix_size;
    size_t optimal_chunk;
    
    if (matrix_size <= 1000) optimal_chunk = matrix_size;
    else if (matrix_size <= 10000) optimal_chunk = 1000;
    else if (matrix_size <= 50000) optimal_chunk = 512;
    else optimal_chunk = H5_MIN_CHUNK_SIZE;
    if (optimal_chunk < H5_MIN_CHUNK_SIZE) optimal_chunk = H5_MIN_CHUNK_SIZE;
    else if (optimal_chunk > H5_MAX_CHUNK_SIZE) optimal_chunk = H5_MAX_CHUNK_SIZE;
    
    handler->chunk_dims[0] = optimal_chunk;
    handler->chunk_dims[1] = optimal_chunk;
    
    print_verbose("HDF5 chunk size: %zu x %zu for %zu x %zu matrix", optimal_chunk, optimal_chunk, matrix_size, matrix_size);
}

INLINE bool setup_hdf5_file(H5Handler* handler) {
    if (!get_mode_write()) return true;

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    
    handler->file_id = H5Fcreate(get_output_file_path(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    
    if (handler->file_id < 0) {
        print_error("Failed to create HDF5 file: %s", get_output_file_path());
        return false;
    }
    
    handler->matrix_dims[0] = handler->matrix_size;
    handler->matrix_dims[1] = handler->matrix_size;
    hid_t matrix_space = H5Screate_simple(2, handler->matrix_dims, NULL);
    
    if (matrix_space < 0) {
        print_error("Failed to create matrix dataspace");
        H5Fclose(handler->file_id);
        return false;
    }
    
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, 2, handler->chunk_dims);
    
    int compression_level = get_compression_level();
    if (compression_level > 0) {
        H5Pset_deflate(plist_id, compression_level);
    }
    
    handler->matrix_dataset_id = H5Dcreate2(
        handler->file_id, 
        "/similarity_matrix", 
        H5T_STD_I32LE, 
        matrix_space, 
        H5P_DEFAULT,
        plist_id,
        H5P_DEFAULT
    );
    
    H5Sclose(matrix_space);
    
    if (handler->matrix_dataset_id < 0) {
        print_error("Failed to create similarity matrix dataset");
        H5Pclose(plist_id);
        H5Fclose(handler->file_id);
        return false;
    }
    
    hid_t seq_group = H5Gcreate2(
        handler->file_id, 
        "/sequences", 
        H5P_DEFAULT,
        H5P_DEFAULT, 
        H5P_DEFAULT
    );
    
    if (seq_group < 0) {
        print_error("Failed to create sequences group");
        H5Dclose(handler->matrix_dataset_id);
        H5Pclose(plist_id);
        H5Fclose(handler->file_id);
        return false;
    }
    
    H5Gclose(seq_group);
    
    handler->seq_dims[0] = handler->matrix_size;
    hid_t seq_lengths_space = H5Screate_simple(1, handler->seq_dims, NULL);
    
    handler->seq_lengths_dataset_id = H5Dcreate2(
        handler->file_id,
        "/sequences/lengths",
        H5T_STD_U64LE,
        seq_lengths_space,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT
    );
    
    H5Sclose(seq_lengths_space);
    H5Pclose(plist_id);
    
    if (handler->seq_lengths_dataset_id < 0) {
        print_error("Failed to create sequence lengths dataset");
        H5Dclose(handler->matrix_dataset_id);
        H5Fclose(handler->file_id);
        return false;
    }
    
    return true;
}

INLINE H5Handler init_h5_handler(size_t matrix_size) {
    H5Handler handler = {0};
    
    handler.matrix_size = matrix_size;
    handler.matrix_checksum = 0;
    handler.sequences_stored = false;
    handler.is_init = false;
    
    if (get_mode_write()) {
        if (!init_matrix_buffer(&handler.buffer, matrix_size)) {
            print_error("Failed to initialize matrix buffer");
            return handler;
        }
        
        print_verbose("Initialized matrix buffer (%zu KB)", (matrix_size * matrix_size * sizeof(int)) / 1024);
    }
    
    calculate_chunk_dimensions(&handler);
    
    if (!get_mode_write()) {
        handler.is_init = true;
        return handler;
    }
    
    if (!setup_hdf5_file(&handler)) {
        free_matrix_buffer(&handler.buffer);
        return handler;
    }
    
    handler.is_init = true;
    print_verbose("H5 handler initialized with matrix size: %zu x %zu", matrix_size, matrix_size);
    
    return handler;
}

INLINE bool flush_matrix_to_hdf5(H5Handler* handler) {
    if (!get_mode_write() || !handler->is_init) return true;
    
    double write_start = get_time();
    
    herr_t status = H5Dwrite(
        handler->matrix_dataset_id, 
        H5T_NATIVE_INT, 
        H5S_ALL, 
        H5S_ALL, 
        H5P_DEFAULT, 
        handler->buffer.data
    );
    
    if (get_mode_benchmark()) {
        add_io_time(get_time() - write_start);
    }
    
    if (status < 0) {
        print_error("Failed to write matrix to HDF5 file");
        return false;
    }
    
    return true;
}

INLINE void set_matrix_value(H5Handler* handler, size_t row, size_t col, int value) {
    if (!get_mode_write() || !handler->is_init || row >= handler->matrix_size || col >= handler->matrix_size) return;
    
    size_t pos1 = row * handler->matrix_size + col;
    handler->buffer.data[pos1] = value;
    
    if (row != col) {
        size_t pos2 = col * handler->matrix_size + row;
        handler->buffer.data[pos2] = value;
    }
}

INLINE void store_sequences_in_h5(H5Handler* handler, Sequence* sequences, size_t count) {
    if (!get_mode_write() || !handler->is_init || handler->sequences_stored) return;

    double write_start = get_time();
    
    print_verbose("Storing %zu sequences in HDF5 file", count);
    
    handler->max_seq_length = 0;
    for (size_t i = 0; i < count; i++) {
        if (sequences[i].length > handler->max_seq_length) {
            handler->max_seq_length = sequences[i].length;
        }
    }
    
    hid_t string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, H5T_VARIABLE);
    
    hid_t seq_space = H5Screate_simple(1, handler->seq_dims, NULL);
    
    if (seq_space < 0) {
        print_error("Failed to create sequence dataspace");
        H5Tclose(string_type);
        return;
    }
    
    handler->seq_dataset_id = H5Dcreate2(
        handler->file_id,
        "/sequences/data",
        string_type,
        seq_space,
        H5P_DEFAULT,
        H5P_DEFAULT,
        H5P_DEFAULT
    );
    
    if (handler->seq_dataset_id < 0) {
        print_error("Failed to create sequences dataset");
        H5Sclose(seq_space);
        H5Tclose(string_type);
        return;
    }
    
    size_t* lengths = (size_t*)malloc(count * sizeof(size_t));
    if (!lengths) {
        print_error("Failed to allocate memory for sequence lengths");
        H5Sclose(seq_space);
        H5Tclose(string_type);
        H5Dclose(handler->seq_dataset_id);
        return;
    }
    
    for (size_t i = 0; i < count; i++) {
        lengths[i] = sequences[i].length;
    }
    
    herr_t status = H5Dwrite(
        handler->seq_lengths_dataset_id,
        H5T_NATIVE_ULONG,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        lengths
    );
    
    free(lengths);
    
    if (status < 0) {
        print_error("Failed to write sequence lengths");
        H5Sclose(seq_space);
        H5Tclose(string_type);
        H5Dclose(handler->seq_dataset_id);
        return;
    }
    
    const size_t batch_size = H5_SEQUENCE_BATCH_SIZE;
    
    for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
        size_t batch_end = batch_start + batch_size;
        if (batch_end > count) batch_end = count;
        size_t current_batch_size = batch_end - batch_start;
        
        char** seq_data = (char**)malloc(current_batch_size * sizeof(char*));
        if (!seq_data) {
            print_error("Failed to allocate memory for sequence batch");
            H5Sclose(seq_space);
            H5Tclose(string_type);
            H5Dclose(handler->seq_dataset_id);
            return;
        }
        
        for (size_t i = 0; i < current_batch_size; i++) {
            seq_data[i] = sequences[batch_start + i].data;
        }
        
        hsize_t batch_dims[1] = {current_batch_size};
        hid_t batch_mem_space = H5Screate_simple(1, batch_dims, NULL);
        
        if (batch_mem_space < 0) {
            print_error("Failed to create memory dataspace for sequence batch");
            free(seq_data);
            H5Sclose(seq_space);
            H5Tclose(string_type);
            H5Dclose(handler->seq_dataset_id);
            return;
        }
        
        hsize_t start[1] = {batch_start};
        hsize_t count[1] = {current_batch_size};
        status = H5Sselect_hyperslab(seq_space, H5S_SELECT_SET, start, NULL, count, NULL);
        
        if (status < 0) {
            print_error("Failed to select hyperslab for sequence batch");
            H5Sclose(batch_mem_space);
            free(seq_data);
            H5Sclose(seq_space);
            H5Tclose(string_type);
            H5Dclose(handler->seq_dataset_id);
            return;
        }
        
        status = H5Dwrite(
            handler->seq_dataset_id,
            string_type,
            batch_mem_space,
            seq_space,
            H5P_DEFAULT,
            seq_data
        );
        
        H5Sclose(batch_mem_space);
        free(seq_data);
        
        if (status < 0) {
            print_error("Failed to write sequence batch");
            H5Sclose(seq_space);
            H5Tclose(string_type);
            H5Dclose(handler->seq_dataset_id);
            return;
        }
        
        if (batch_start > 0 && batch_start % 10000 == 0) {
            print_verbose("  Stored %zu/%zu sequences...", batch_start, handler->matrix_size);
        }
    }
    
    H5Sclose(seq_space);
    H5Tclose(string_type);
    
    if (get_mode_benchmark()) {
        add_io_time(get_time() - write_start);
    }
    
    handler->sequences_stored = true;
    
    print_verbose("Successfully stored sequences in HDF5 file");
}

INLINE int64_t calculate_matrix_checksum(H5Handler* handler) {
    if (!handler->is_init) {
        print_error("Cannot calculate checksum: H5Handler not initialized");
        return 0;
    }
    
    int64_t checksum = 0;
    size_t size = handler->matrix_size;
    
    if (get_mode_write()) {
        for (size_t i = 0; i < size; i++) {
            for (size_t j = i; j < size; j++) {
                size_t pos = i * size + j;
                int value = handler->buffer.data[pos];
                
                if (i == j) {
                    checksum += value;
                } else {
                    checksum += value * 2;
                }
            }
        }
    }
    
    handler->matrix_checksum = checksum;
    return checksum;
}

INLINE void close_h5_handler(H5Handler* handler) {
    if (!handler->is_init) return;
    
    if (get_mode_write()) {
        print_step_header("Finalizing Results");
        print_info("Writing results to output file: %s", get_file_name(get_output_file_path()));
        flush_matrix_to_hdf5(handler);
        int64_t checksum = calculate_matrix_checksum(handler);
        print_info("Matrix checksum: %lld", checksum);
        bench_write_end();

        if (handler->seq_dataset_id > 0) H5Dclose(handler->seq_dataset_id);
        if (handler->seq_lengths_dataset_id > 0) H5Dclose(handler->seq_lengths_dataset_id);
        if (handler->matrix_dataset_id > 0) H5Dclose(handler->matrix_dataset_id);
        if (handler->file_id > 0) H5Fclose(handler->file_id);
        free_matrix_buffer(&handler->buffer);
    }
    
    handler->is_init = false;
}

#endif // H5_HANDLER_H