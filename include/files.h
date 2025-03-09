#ifndef FILES_H
#define FILES_H

#include "seqalign.h"
#include "print.h"

typedef struct {
    char* file_data;
    size_t data_size;
    #ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
    #else
    int fd;
    #endif
} File;

INLINE File get_input_file(void) {
    File file = {0};
    const char* file_path = get_input_file_path();

    #ifdef _WIN32
    file.hFile = CreateFileA(file_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (file.hFile == INVALID_HANDLE_VALUE) {
        print_error("Could not open input file '%s'", file_path);
        print_step_header_end();
        exit(1);
    }
    file.hMapping = CreateFileMapping(file.hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (file.hMapping == NULL) {
        print_error("Could not create file mapping for '%s'", file_path);
        print_step_header_end();
        CloseHandle(file.hFile);
        exit(1);
    }
    file.file_data = (char*)MapViewOfFile(file.hMapping, FILE_MAP_READ, 0, 0, 0);
    if (file.file_data == NULL) {
        print_error("Could not map view of file '%s'", file_path);
        print_step_header_end();
        CloseHandle(file.hMapping);
        CloseHandle(file.hFile);
        exit(1);
    }
    LARGE_INTEGER file_size;
    GetFileSizeEx(file.hFile, &file_size);
    file.data_size = file_size.QuadPart;
    #else
    file.fd = open(file_path, O_RDONLY);
    if (file.fd == -1) {
        print_error("Could not open input file '%s'", file_path);
        print_step_header_end();
        exit(1);
    }
    struct stat sb;
    if (fstat(file.fd, &sb) == -1) {
        print_error("Could not stat file '%s'", file_path);
        print_step_header_end();
        close(file.fd);
        exit(1);
    }
    file.data_size = sb.st_size;
    file.file_data = mmap(NULL, file.data_size, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (file.file_data == MAP_FAILED) {
        print_error("Could not memory map file '%s'", file_path);
        print_step_header_end();
        close(file.fd);
        exit(1);
    }
    madvise(file.file_data, file.data_size, MADV_SEQUENTIAL);
    #endif
    return file;
}

INLINE void free_input_file(File* file) {
    #ifdef _WIN32
    UnmapViewOfFile(file->file_data);
    CloseHandle(file->hMapping);
    CloseHandle(file->hFile);
    #else
    munmap(file->file_data, file->data_size);
    close(file->fd);
    #endif
}

#endif