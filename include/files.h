#ifndef FILES_H
#define FILES_H

#include "seqalign.h"

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

    #ifdef _WIN32
    file.hFile = CreateFileA(get_input_file_path(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    file.hMapping = CreateFileMapping(file.hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    file.file_data = (char*)MapViewOfFile(file.hMapping, FILE_MAP_READ, 0, 0, 0);
    LARGE_INTEGER file_size;
    GetFileSizeEx(file.hFile, &file_size);
    file.data_size = file_size.QuadPart;
    #else
    file.fd = open(get_input_file_path(), O_RDONLY);
    struct stat sb;
    fstat(file.fd, &sb);
    file.data_size = sb.st_size;
    file.file_data = mmap(NULL, file.data_size, PROT_READ, MAP_PRIVATE, file.fd, 0);
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