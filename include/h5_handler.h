#ifndef H5_HANDLER_H
#define H5_HANDLER_H

#include "common.h"
#include "benchmark.h"
#include "sequence.h"
#include "files.h"
#include <hdf5.h>

#define H5_MIN_CHUNK_SIZE 128
#define H5_MAX_CHUNK_SIZE 1024
#define H5_SEQUENCE_BATCH_SIZE 5000
#define MMAP_MEMORY_USAGE_THRESHOLD 0.7

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
    MmapMatrix mmap_matrix;
    bool use_mmap;
    char mmap_filename[MAX_PATH];
    
    int64_t* thread_checksums;
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
        handler->file_id = -1;
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
        handler->file_id = -1;
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
        handler->matrix_dataset_id = -1;
        H5Pclose(plist_id);
        H5Fclose(handler->file_id);
        handler->file_id = -1;
        return false;
    }
    
    H5Gclose(seq_group);
    
    handler->seq_dims[0] = handler->matrix_size;
    hid_t seq_lengths_space = H5Screate_simple(1, handler->seq_dims, NULL);
    
    if (seq_lengths_space < 0) {
        print_error("Failed to create sequence lengths dataspace");
        H5Dclose(handler->matrix_dataset_id);
        handler->matrix_dataset_id = -1;
        H5Pclose(plist_id);
        H5Fclose(handler->file_id);
        handler->file_id = -1;
        return false;
    }
    
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
        handler->matrix_dataset_id = -1;
        H5Fclose(handler->file_id);
        handler->file_id = -1;
        return false;
    }
    
    return true;
}

INLINE H5Handler init_h5_handler(size_t matrix_size) {
    H5Handler handler = {0};
    
    handler.matrix_size = matrix_size;
    handler.matrix_checksum = 0;
    handler.thread_checksums = NULL;
    handler.sequences_stored = false;
    handler.is_init = false;
    handler.use_mmap = false;
    handler.file_id = -1;
    handler.matrix_dataset_id = -1;
    handler.seq_dataset_id = -1;
    handler.seq_lengths_dataset_id = -1;
    
    if (get_mode_write()) {
        handler.use_mmap = check_matrix_exceeds_memory(matrix_size, MMAP_MEMORY_USAGE_THRESHOLD);
        
        if (handler.use_mmap) {
            get_mmap_matrix_filename(handler.mmap_filename, MAX_PATH, get_output_file_path());
            print_info("Matrix size exceeds memory threshold, using memory-mapped file");
            handler.mmap_matrix = create_mmap_matrix(handler.mmap_filename, matrix_size);
            if (!handler.mmap_matrix.data) {
                print_error("Failed to create memory-mapped matrix");
                return handler;
            }
        } else {
            if (!init_matrix_buffer(&handler.buffer, matrix_size)) {
                print_error("Failed to initialize matrix buffer");
                return handler;
            }
        }
        
        calculate_chunk_dimensions(&handler);
        
        if (!setup_hdf5_file(&handler)) {
            if (!handler.use_mmap) {
                free_matrix_buffer(&handler.buffer);
            } else {
                close_mmap_matrix(&handler.mmap_matrix);
                remove(handler.mmap_filename);
            }
            return handler;
        }
        
        if (get_mode_multithread()) {
            handler.thread_checksums = (int64_t*)aligned_alloc(CACHE_LINE, sizeof(int64_t) * get_num_threads());
            if (!handler.thread_checksums) {
                print_error("Failed to allocate memory for thread checksums");
                if (!handler.use_mmap) {
                    free_matrix_buffer(&handler.buffer);
                } else {
                    close_mmap_matrix(&handler.mmap_matrix);
                    remove(handler.mmap_filename);
                }
                if (handler.matrix_dataset_id > 0) H5Dclose(handler.matrix_dataset_id);
                if (handler.seq_lengths_dataset_id > 0) H5Dclose(handler.seq_lengths_dataset_id);
                if (handler.file_id > 0) H5Fclose(handler.file_id);
                return handler;
            }
            memset(handler.thread_checksums, 0, sizeof(int64_t) * get_num_threads());
        }
        
        handler.is_init = true;
        print_verbose("%s file created with matrix size: %zu x %zu", handler.use_mmap ? "Memory-mapped" : "HDF5", matrix_size, matrix_size);
    } else {
        handler.is_init = true;
    }
    
    return handler;
}

INLINE bool flush_matrix_to_hdf5(H5Handler* handler) {
    if (!get_mode_write() || !handler->is_init) return true;
    
    if (handler->matrix_dataset_id < 0 || (handler->use_mmap && !handler->mmap_matrix.data) || (!handler->use_mmap && !handler->buffer.data)) {
        print_error("Cannot flush matrix: HDF5 resources not properly initialized");
        return false;
    }

    double write_start = get_time();
    herr_t status = -1;
    
    if (handler->use_mmap) {
        size_t matrix_size = handler->matrix_size;
        
        size_t available_mem = get_available_memory();
        if (available_mem == 0) {
            print_error("Failed to retrieve available memory");
            return false;
        }

        size_t row_bytes = matrix_size * sizeof(int);
        size_t max_rows = available_mem / (4 * row_bytes);
        size_t chunk_size = (max_rows < 4) ? 4 : max_rows;
        if (chunk_size > 1024) chunk_size = 1024;
        
        print_verbose("Converting matrix using %zu rows per chunk (%zu MiB buffer)", chunk_size, (chunk_size * row_bytes) / MiB);
        
        int* buffer = (int*)malloc(chunk_size * row_bytes);
        if (!buffer) {
            print_warning("Failed to allocate transfer buffer of %zu bytes", chunk_size * row_bytes);
            chunk_size = 1;
            buffer = (int*)malloc(chunk_size * row_bytes);
            if (!buffer) {
                print_error("Cannot allocate even minimal buffer, aborting");
                return false;
            }
            print_warning("Using minimal buffer size of 1 row (%zu bytes)", row_bytes);
        }
        
        hid_t file_space = H5Dget_space(handler->matrix_dataset_id);
        if (file_space < 0) {
            free(buffer);
            return false;
        }
        
        print_info("Converting memory-mapped matrix to HDF5 format");
        
        for (size_t start_row = 0; start_row < matrix_size; start_row += chunk_size) {
            size_t end_row = start_row + chunk_size;
            if (end_row > matrix_size) end_row = matrix_size;
            size_t rows = end_row - start_row;
            
            memset(buffer, 0, rows * row_bytes);
            
            for (size_t i = start_row; i < end_row; i++) {
                for (size_t j = i; j < matrix_size; j++) {
                    size_t idx = mmap_triangle_index(i, j, matrix_size);
                    int value = handler->mmap_matrix.data[idx];
                    buffer[(i - start_row) * matrix_size + j] = value;
                }
                
                for (size_t j = 0; j < i; j++) {
                    if (j >= start_row) {
                        buffer[(i - start_row) * matrix_size + j] = buffer[(j - start_row) * matrix_size + i];
                    } else {
                        size_t idx = mmap_triangle_index(j, i, matrix_size);
                        int value = handler->mmap_matrix.data[idx];
                        buffer[(i - start_row) * matrix_size + j] = value;
                    }
                }
            }
            
            hsize_t start[2] = {start_row, 0};
            hsize_t count[2] = {rows, matrix_size};
            H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
            
            hsize_t mem_dims[2] = {rows, matrix_size};
            hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);
            
            if (mem_space < 0) {
                print_error("Failed to create memory dataspace for matrix chunk");
                H5Sclose(file_space);
                free(buffer);
                return false;
            }
            
            status = H5Dwrite(handler->matrix_dataset_id, H5T_NATIVE_INT, mem_space, file_space, H5P_DEFAULT, buffer);
            
            H5Sclose(mem_space);
            
            if (status < 0) {
                print_progress_bar_end();
                print_error("Failed to write chunk to HDF5");
                H5Sclose(file_space);
                free(buffer);
                return false;
            }
            
            print_progress_bar((double)end_row / matrix_size, 40, "Converting to HDF5");
        }
        
        print_progress_bar_end();
        
        H5Sclose(file_space);
        free(buffer);
    } else {
        status = H5Dwrite(
            handler->matrix_dataset_id, 
            H5T_NATIVE_INT, 
            H5S_ALL, 
            H5S_ALL, 
            H5P_DEFAULT, 
            handler->buffer.data
        );
        
        if (status < 0) {
            print_error("Failed to write matrix data to HDF5");
            return false;
        }
    }
    
    if (get_mode_benchmark()) {
        add_io_time(get_time() - write_start);
    }
    
    return true;
}

INLINE void set_matrix_value(H5Handler* handler, size_t row, size_t col, int value) {
    if (!get_mode_write() || !handler->is_init || row >= handler->matrix_size || col >= handler->matrix_size) return;
    
    if (handler->use_mmap) {
        if (row > col) return;
        if (!handler->mmap_matrix.data) return;
        
        mmap_set_matrix_value(&handler->mmap_matrix, row, col, value);
    } else {
        if (!handler->buffer.data) return;
        
        size_t pos1 = row * handler->matrix_size + col;
        handler->buffer.data[pos1] = value;
        
        if (row != col) {
            size_t pos2 = col * handler->matrix_size + row;
            handler->buffer.data[pos2] = value;
        }
    }
}

INLINE bool store_sequences_in_h5(H5Handler* handler, Sequence* sequences, size_t count) {
    if (!get_mode_write() || !handler->is_init || handler->sequences_stored) return true;
    
    if (handler->file_id < 0) {
        print_error("Cannot store sequences: HDF5 file not initialized");
        return false;
    }

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
        return false;
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
        return false;
    }
    
    size_t* lengths = (size_t*)malloc(count * sizeof(size_t));
    if (!lengths) {
        print_error("Failed to allocate memory for sequence lengths");
        H5Sclose(seq_space);
        H5Tclose(string_type);
        H5Dclose(handler->seq_dataset_id);
        handler->seq_dataset_id = -1;
        return false;
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
        handler->seq_dataset_id = -1;
        return false;
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
            handler->seq_dataset_id = -1;
            return false;
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
            handler->seq_dataset_id = -1;
            return false;
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
            handler->seq_dataset_id = -1;
            return false;
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
            handler->seq_dataset_id = -1;
            return false;
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
    return true;
}

INLINE int64_t collect_thread_checksums(H5Handler* handler) {
    if (!handler->thread_checksums) return 0;
    
    int64_t total_checksum = 0;
    for (int t = 0; t < get_num_threads(); t++) {
        total_checksum += handler->thread_checksums[t];
    }
    return total_checksum * 2;
}

INLINE void close_h5_handler(H5Handler* handler) {
    if (!handler->is_init) return;
    
    if (get_mode_write()) {
        bool success = true;
        
        print_step_header("Finalizing Results");
        print_info("Writing results to output file: %s", get_file_name(get_output_file_path()));
        
        if (!flush_matrix_to_hdf5(handler)) {
            print_error("Failed to write matrix data to output file");
            success = false;
        }
        
        if (get_mode_multithread() && handler->thread_checksums) {
            handler->matrix_checksum = collect_thread_checksums(handler);
            aligned_free(handler->thread_checksums);
            handler->thread_checksums = NULL;
        }
        
        print_info("Matrix checksum: %lld", handler->matrix_checksum);
        bench_write_end();

        if (handler->seq_dataset_id > 0) {
            H5Dclose(handler->seq_dataset_id);
            handler->seq_dataset_id = -1;
        }
        
        if (handler->seq_lengths_dataset_id > 0) {
            H5Dclose(handler->seq_lengths_dataset_id);
            handler->seq_lengths_dataset_id = -1;
        }
        
        if (handler->matrix_dataset_id > 0) {
            H5Dclose(handler->matrix_dataset_id);
            handler->matrix_dataset_id = -1;
        }
        
        if (handler->file_id > 0) {
            H5Fclose(handler->file_id);
            handler->file_id = -1;
        }
        
        if (handler->use_mmap) {
            close_mmap_matrix(&handler->mmap_matrix);
            if (success) {
                if (remove(handler->mmap_filename) != 0) {
                    print_warning("Failed to remove temporary file: %s", handler->mmap_filename);
                }
            } else {
                print_warning("Keeping temporary file for debugging: %s", handler->mmap_filename);
            }
        } else {
            free_matrix_buffer(&handler->buffer);
        }
    }
    
    handler->is_init = false;
}

#endif // H5_HANDLER_H