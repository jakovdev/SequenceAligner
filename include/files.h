#ifndef FILES_H
#define FILES_H

#include "arch.h"
#include "print.h"

typedef struct
{
    char* data;
    size_t size;
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#else
    int fd;
#endif
} File;

typedef struct
{
    int* data;
    size_t matrix_size;
    size_t file_size;
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#else
    int fd;
#endif
} MmapMatrix;

INLINE const char*
file_name_path(const char* path)
{
#ifdef _WIN32
    const char* name = strrchr(path, '\\');
#else
    const char* name = strrchr(path, '/');
#endif
    return name ? name + 1 : path;
}

INLINE void
file_read(File* file, const char* file_path)
{
    const char* file_name = file_name_path(file_path);

#ifdef _WIN32
    file->hFile = CreateFileA(file_path,
                              GENERIC_READ,
                              FILE_SHARE_READ,
                              NULL,
                              OPEN_EXISTING,
                              FILE_FLAG_SEQUENTIAL_SCAN,
                              NULL);

    if (file->hFile == INVALID_HANDLE_VALUE)
    {
        print(ERROR, MSG_NONE, "Could not open file '%s'", file_name);
        exit(1);
    }

    file->hMapping = CreateFileMapping(file->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (file->hMapping == NULL)
    {
        print(ERROR, MSG_NONE, "Could not create file mapping for '%s'", file_name);
        CloseHandle(file->hFile);
        exit(1);
    }

    file->file_data = (char*)MapViewOfFile(file->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (file->file_data == NULL)
    {
        print(ERROR, MSG_NONE, "Could not map view of file '%s'", file_name);
        CloseHandle(file->hMapping);
        CloseHandle(file->hFile);
        exit(1);
    }

    LARGE_INTEGER file_size;
    GetFileSizeEx(file->hFile, &file_size);
    file->data_size = file_size.QuadPart;
#else
    file->fd = open(file_path, O_RDONLY);
    if (file->fd == -1)
    {
        print(ERROR, MSG_NONE, "Could not open input file '%s'", file_name);
        exit(1);
    }

    struct stat sb;
    if (fstat(file->fd, &sb) == -1)
    {
        print(ERROR, MSG_NONE, "Could not stat file '%s'", file_name);
        close(file->fd);
        exit(1);
    }

    file->size = sb.st_size;
    file->data = mmap(NULL, file->size, PROT_READ, MAP_PRIVATE, file->fd, 0);
    if (file->data == MAP_FAILED)
    {
        print(ERROR, MSG_NONE, "Could not memory map file '%s'", file_name);
        close(file->fd);
        exit(1);
    }

    madvise(file->data, file->size, MADV_SEQUENTIAL);
#endif
}

INLINE void
file_free(File* file)
{
#ifdef _WIN32
    UnmapViewOfFile(file->file_data);
    CloseHandle(file->hMapping);
    CloseHandle(file->hFile);
#else
    munmap(file->data, file->size);
    close(file->fd);
#endif
}

INLINE MmapMatrix
mmap_matrix_create(const char* file_path, size_t matrix_size)
{
    MmapMatrix matrix = { 0 };
    matrix.matrix_size = matrix_size;

    size_t triangle_elements = (matrix_size * (matrix_size + 1)) / 2;
    size_t bytes_needed = triangle_elements * sizeof(int);
    matrix.file_size = bytes_needed;
    const char* file_name = file_name_path(file_path);

    print(INFO,
          MSG_LOC(LAST),
          "Creating memory-mapped matrix file: %s (%.2f GiB)",
          file_name,
          bytes_needed / (float)GiB);

#ifdef _WIN32
    matrix.hFile = CreateFileA(file_path,
                               GENERIC_READ | GENERIC_WRITE,
                               0,
                               NULL,
                               CREATE_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL,
                               NULL);

    if (matrix.hFile == INVALID_HANDLE_VALUE)
    {
        print(ERROR, MSG_NONE, "Could not create memory-mapped file '%s'", file_name);
        return matrix;
    }

    LARGE_INTEGER file_size;
    file_size.QuadPart = bytes_needed;
    SetFilePointerEx(matrix.hFile, file_size, NULL, FILE_BEGIN);
    SetEndOfFile(matrix.hFile);

    matrix.hMapping = CreateFileMapping(matrix.hFile, NULL, PAGE_READWRITE, 0, 0, NULL);
    if (matrix.hMapping == NULL)
    {
        print(ERROR, MSG_NONE, "Could not create file mapping for '%s'", file_name);
        CloseHandle(matrix.hFile);
        matrix.hFile = INVALID_HANDLE_VALUE;
        return matrix;
    }

    matrix.data = (int*)MapViewOfFile(matrix.hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (matrix.data == NULL)
    {
        print(ERROR, MSG_NONE, "Could not map view of file '%s'", file_name);
        CloseHandle(matrix.hMapping);
        CloseHandle(matrix.hFile);
        matrix.hMapping = NULL;
        matrix.hFile = INVALID_HANDLE_VALUE;
        return matrix;
    }

#else
    matrix.fd = open(file_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (matrix.fd == -1)
    {
        print(ERROR, MSG_NONE, "Could not create memory-mapped file '%s'", file_name);
        return matrix;
    }

    if (ftruncate(matrix.fd, bytes_needed) == -1)
    {
        print(ERROR, MSG_NONE, "Could not set size for file '%s'", file_name);
        close(matrix.fd);
        matrix.fd = -1;
        return matrix;
    }

    matrix.data = (int*)mmap(NULL, bytes_needed, PROT_READ | PROT_WRITE, MAP_SHARED, matrix.fd, 0);
    if (matrix.data == MAP_FAILED)
    {
        print(ERROR, MSG_NONE, "Could not memory map file '%s'", file_name);
        close(matrix.fd);
        matrix.fd = -1;
        return matrix;
    }

    madvise(matrix.data, bytes_needed, MADV_RANDOM);
    madvise(matrix.data, bytes_needed, MADV_HUGEPAGE);
    madvise(matrix.data, bytes_needed, MADV_DONTFORK);

#endif

    size_t check_count = 5;
    size_t total_elements = bytes_needed / sizeof(int);
    size_t check_indices[5] = {
        0,                      // First element
        total_elements / 4,     // 25% in
        total_elements / 2,     // Middle
        total_elements * 3 / 4, // 75% in
        total_elements - 1      // Last element
    };

    bool is_zeroed = true;
    for (size_t i = 0; i < check_count; i++)
    {
        if (matrix.data[check_indices[i]] != 0)
        {
            is_zeroed = false;
            break;
        }
    }

    if (!is_zeroed)
    {
        print(VERBOSE, MSG_LOC(FIRST), "Memory not pre-zeroed, performing explicit initialization");

        double pre_memset_time = time_current();
        memset(matrix.data, 0, bytes_needed);
        double memset_time = pre_memset_time - time_current();

        print(VERBOSE, MSG_LOC(LAST), "Matrix data memset performed in %.2f seconds", memset_time);
    }

    return matrix;
}

INLINE void
mmap_matrix_close(MmapMatrix* matrix)
{
    if (!matrix->data)
    {
        return;
    }

#ifdef _WIN32
    UnmapViewOfFile(matrix->data);
    CloseHandle(matrix->hMapping);
    CloseHandle(matrix->hFile);
#else
    munmap(matrix->data, matrix->file_size);
    close(matrix->fd);
#endif

    matrix->data = NULL;
    matrix->file_size = 0;
}

INLINE size_t
mmap_triangle_index(size_t row, size_t col, size_t matrix_size)
{
    return (row * matrix_size - (row * (row + 1)) / 2) + col - row;
}

INLINE void
mmap_matrix_set_value(MmapMatrix* matrix, size_t row, size_t col, int value)
{
    if (!matrix->data || row >= matrix->matrix_size || col >= matrix->matrix_size)
    {
        return;
    }

    size_t index = mmap_triangle_index(row, col, matrix->matrix_size);
    matrix->data[index] = value;
}

INLINE void
mmap_matrix_file_name(char* buffer, size_t buffer_size, const char* output_path)
{
    if (output_path && output_path[0] != '\0')
    {
        char dir[MAX_PATH] = { 0 };
        char base[MAX_PATH] = { 0 };

        const char* last_slash = strrchr(output_path, '/');
        if (last_slash)
        {
            size_t dir_len = last_slash - output_path + 1;
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