#ifndef H5_HANDLER_H
#define H5_HANDLER_H

#include "common.h"
#include "benchmark.h"
#include <hdf5.h>

typedef struct {
    int* data;
    size_t size;
    size_t capacity;
} MatrixBuffer;

typedef struct {
    hid_t file_id;
    hid_t dataset_id;
    hid_t dataspace_id;
    hsize_t dim_sizes[2];
    size_t matrix_size;
    MatrixBuffer buffer;
} H5Handler;

INLINE void init_matrix_buffer(MatrixBuffer* buffer, size_t matrix_size) {
    buffer->size = matrix_size;
    buffer->capacity = matrix_size;
    buffer->data = (int*)calloc(matrix_size * matrix_size, sizeof(int));
}

INLINE H5Handler init_h5_handler(size_t matrix_size) {
    H5Handler handler = {0};
    handler.matrix_size = matrix_size;
    
    init_matrix_buffer(&handler.buffer, matrix_size);
    if (!get_mode_write()) return handler;
    
    handler.file_id = H5Fcreate(get_output_file_path(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    handler.dim_sizes[0] = matrix_size;
    handler.dim_sizes[1] = matrix_size;
    handler.dataspace_id = H5Screate_simple(2, handler.dim_sizes, NULL);
    
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    size_t chunk_dim = (size_t)sqrt(((4.0 * KiB) / sizeof(int)) * sqrt(matrix_size / 1000.0));
    chunk_dim = chunk_dim < 16 ? 16 : (chunk_dim > 128 ? 128 : chunk_dim);
    chunk_dim = chunk_dim > matrix_size ? matrix_size : chunk_dim;
    print_verbose("HDF5 chunk size for %zux%zu matrix: %zu", matrix_size, matrix_size, chunk_dim);
    hsize_t chunk_dims[2] = {chunk_dim, chunk_dim};
    H5Pset_chunk(plist_id, 2, chunk_dims);
    H5Pset_deflate(plist_id, get_compression_level());
    
    handler.dataset_id = H5Dcreate2(handler.file_id, "similarity_matrix", 
                                    H5T_STD_I32LE, handler.dataspace_id, 
                                    H5P_DEFAULT, plist_id, H5P_DEFAULT);
    
    H5Pclose(plist_id);
    
    return handler;
}

INLINE void set_matrix_value(H5Handler* restrict handler, size_t row, size_t col, int value) {
    handler->buffer.data[row * handler->matrix_size + col] = value;
}

INLINE void close_h5_handler(H5Handler* restrict handler) {
    if (get_mode_write()) {
        print_info("Writing results to output file: %s", get_file_name(get_output_file_path()));
        bench_write_start();
        H5Dwrite(handler->dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, handler->buffer.data);
        H5Fflush(handler->file_id, H5F_SCOPE_LOCAL);
        H5Dclose(handler->dataset_id);
        H5Sclose(handler->dataspace_id);
        H5Fclose(handler->file_id);
        bench_write_end();
    }
    free(handler->buffer.data);
}

#endif