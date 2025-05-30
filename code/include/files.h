#pragma once
#ifndef FILES_H
#define FILES_H

#include "arch.h"
#include "print.h"

typedef struct
{
    size_t bytes;
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#else
    int fd;
#endif
} FileMetadata;

typedef struct
{
    FileMetadata meta;
    char* text;
} FileText;

typedef struct
{
    FileMetadata meta;
    int* matrix;
} FileMatrix;

static inline void
file_metadata_init(FileMetadata* meta)
{
#ifdef _WIN32
    meta->hFile = INVALID_HANDLE_VALUE;
    meta->hMapping = NULL;
#else
    meta->fd = -1;
#endif
    meta->bytes = 0;
}

static inline void
file_metadata_close(FileMetadata* meta)
{
#ifdef _WIN32
    if (meta->hMapping)
    {
        CloseHandle(meta->hMapping);
        meta->hMapping = NULL;
    }

    if (meta->hFile != INVALID_HANDLE_VALUE)
    {
        CloseHandle(meta->hFile);
        meta->hFile = INVALID_HANDLE_VALUE;
    }

#else
    if (meta->fd != -1)
    {
        close(meta->fd);
        meta->fd = -1;
    }

#endif
    meta->bytes = 0;
}

static inline void
file_text_open(FileText* file, const char* file_path)
{
    file_metadata_init(&file->meta);
    file->text = NULL;

    const char* file_name = file_name_path(file_path);

#ifdef _WIN32
    file->meta.hFile = CreateFileA(file_path,
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   NULL,
                                   OPEN_EXISTING,
                                   FILE_FLAG_SEQUENTIAL_SCAN,
                                   NULL);

    if (file->meta.hFile == INVALID_HANDLE_VALUE)
    {
        print(ERROR, MSG_NONE, "FILE | Could not open file '%s'", file_name);
        return;
    }

    file->meta.hMapping = CreateFileMapping(file->meta.hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (file->meta.hMapping == NULL)
    {
        print(ERROR, MSG_NONE, "FILE | Could not create file mapping for '%s'", file_name);
        file_metadata_close(&file->meta);
        return;
    }

    file->text = (char*)MapViewOfFile(file->meta.hMapping, FILE_MAP_READ, 0, 0, 0);
    if (file->text == NULL)
    {
        print(ERROR, MSG_NONE, "FILE | Could not map view of file '%s'", file_name);
        file_metadata_close(&file->meta);
        return;
    }

    LARGE_INTEGER file_size;
    GetFileSizeEx(file->meta.hFile, &file_size);
    file->meta.bytes = file_size.QuadPart;
#else
    file->meta.fd = open(file_path, O_RDONLY);
    if (file->meta.fd == -1)
    {
        print(ERROR, MSG_NONE, "FILE | Could not open input file '%s'", file_name);
        return;
    }

    struct stat sb;
    if (fstat(file->meta.fd, &sb) == -1)
    {
        print(ERROR, MSG_NONE, "FILE | Could not stat file '%s'", file_name);
        file_metadata_close(&file->meta);
        return;
    }

    file->meta.bytes = (size_t)sb.st_size;
    file->text = (char*)mmap(NULL, file->meta.bytes, PROT_READ, MAP_PRIVATE, file->meta.fd, 0);
    if (file->text == MAP_FAILED)
    {
        print(ERROR, MSG_NONE, "FILE | Could not memory map file '%s'", file_name);
        file_metadata_close(&file->meta);
        file->text = NULL;
        return;
    }

    madvise(file->text, file->meta.bytes, MADV_SEQUENTIAL);
#endif
}

static inline void
file_text_close(FileText* file)
{
#ifdef _WIN32
    if (file->text)
    {
        UnmapViewOfFile(file->text);
        file->text = NULL;
    }

#else
    if (file->text)
    {
        munmap(file->text, file->meta.bytes);
        file->text = NULL;
    }

#endif
    file_metadata_close(&file->meta);
}

static inline FileMatrix
file_matrix_open(const char* file_path, size_t matrix_dim)
{
    FileMatrix file = { 0 };
    file_metadata_init(&file.meta);

    size_t triangle_elements = (matrix_dim * (matrix_dim - 1)) / 2;
    size_t bytes = triangle_elements * sizeof(*file.matrix);
    file.meta.bytes = bytes;
    const char* file_name = file_name_path(file_path);
    const float mmap_size = (float)bytes / (float)GiB;

    print(INFO, MSG_LOC(LAST), "Creating matrix file: %s (%.2f GiB)", file_name, mmap_size);

#ifdef _WIN32
    file.meta.hFile = CreateFileA(file_path,
                                  GENERIC_READ | GENERIC_WRITE,
                                  0,
                                  NULL,
                                  CREATE_ALWAYS,
                                  FILE_ATTRIBUTE_NORMAL,
                                  NULL);

    if (file.meta.hFile == INVALID_HANDLE_VALUE)
    {
        print(ERROR, MSG_NONE, "MATRIXFILE | Could not create memory-mapped file '%s'", file_name);
        return matrix;
    }

    LARGE_INTEGER file_size;
    file_size.QuadPart = bytes;
    SetFilePointerEx(file.meta.hFile, file_size, NULL, FILE_BEGIN);
    SetEndOfFile(file.meta.hFile);

    file.meta.hMapping = CreateFileMapping(file.meta.hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
    if (file.meta.hMapping == NULL)
    {
        print(ERROR, MSG_NONE, "MATRIXFILE | Could not create file mapping for '%s'", file_name);
        file_metadata_close(&file.meta);
        return matrix;
    }

    file.matrix = (int*)MapViewOfFile(file.meta.hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (file.matrix == NULL)
    {
        print(ERROR, MSG_NONE, "MATRIXFILE | Could not map view of file '%s'", file_name);
        file_metadata_close(&file.meta);
        return matrix;
    }

#else
    file.meta.fd = open(file_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (file.meta.fd == -1)
    {
        print(ERROR, MSG_NONE, "MATRIXFILE | Could not create memory-mapped file '%s'", file_name);
        return file;
    }

    if (ftruncate(file.meta.fd, (off_t)bytes) == -1)
    {
        print(ERROR, MSG_NONE, "MATRIXFILE | Could not set size for file '%s'", file_name);
        file_metadata_close(&file.meta);
        return file;
    }

    file.matrix = (int*)mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, file.meta.fd, 0);
    if (file.matrix == MAP_FAILED)
    {
        print(ERROR, MSG_NONE, "MATRIXFILE | Could not memory map file '%s'", file_name);
        file_metadata_close(&file.meta);
        file.matrix = NULL;
        return file;
    }

    madvise(file.matrix, bytes, MADV_RANDOM);
    madvise(file.matrix, bytes, MADV_HUGEPAGE);
    madvise(file.matrix, bytes, MADV_DONTFORK);

#endif

    size_t check_indices[5] = {
        0,                         // First element
        triangle_elements / 4,     // First quarter
        triangle_elements / 2,     // Middle
        triangle_elements * 3 / 4, // Third quarter
        triangle_elements - 1      // Last element
    };

    bool is_zeroed = true;
    for (size_t i = 0; i < (sizeof(check_indices) / sizeof(*check_indices)); i++)
    {
        if (file.matrix[check_indices[i]] != 0)
        {
            is_zeroed = false;
            break;
        }
    }

    if (!is_zeroed)
    {
        print(VERBOSE, MSG_LOC(FIRST), "Memory not pre-zeroed, performing explicit initialization");

        double pre_memset_time = time_current();
        memset(file.matrix, 0, bytes);
        double memset_time = pre_memset_time - time_current();

        print(VERBOSE, MSG_LOC(LAST), "Matrix data memset performed in %.2f seconds", memset_time);
    }

    return file;
}

static inline void
file_matrix_close(FileMatrix* matrix)
{
    if (!matrix->matrix)
    {
        return;
    }

#ifdef _WIN32
    UnmapViewOfFile(matrix->data);
    matrix->data = NULL;
#else
    munmap(matrix->matrix, matrix->meta.bytes);
    matrix->matrix = NULL;
#endif
    file_metadata_close(&matrix->meta);
}

static inline size_t
matrix_triangle_index(size_t row, size_t col)
{
    return (col * (col - 1)) / 2 + row;
}

static inline void
file_matrix_name(char* buffer, size_t buffer_size, const char* output_path)
{
    if (output_path && output_path[0] != '\0')
    {
        char dir[MAX_PATH] = { 0 };
        char base[MAX_PATH] = { 0 };

        const char* last_slash = strrchr(output_path, '/');
        if (last_slash)
        {
            size_t dir_len = (size_t)(last_slash - output_path + 1);
            strncpy(dir, output_path, dir_len);
            dir[dir_len] = '\0';
            strncpy(base, last_slash + 1, MAX_PATH - 1);
        }

        else
        {
            strcpy(dir, "./");
            strncpy(base, output_path, MAX_PATH - 1);
        }

        char* dot = strrchr(base, '.');
        if (dot)
        {
            *dot = '\0';
        }

        snprintf(buffer, buffer_size, "%s%s.mmap", dir, base);
    }

    else
    {
        snprintf(buffer, buffer_size, "./seqalign_matrix.mmap");
    }
}

#endif // FILES_H