#ifndef ARGS_H
#define ARGS_H

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
    #if MODE_WRITE == 1
    WriteBuffer writer;
    #endif
} Files;

INLINE void flush_buffer(WriteBuffer* wb) {
    #ifdef _WIN32
    WriteFile(wb->handle, wb->buffer, (DWORD)wb->pos, &wb->bytes_written, NULL);
    #else
    write(wb->fd, wb->buffer, wb->pos);
    #endif
    wb->pos = 0;
}

INLINE Files get_files(void) {
    Files files = {0};

    #ifdef _WIN32
    files.hFile = CreateFileA(INPUT_FILE, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    files.hMapping = CreateFileMapping(files.hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    files.file_data = (char*)MapViewOfFile(files.hMapping, FILE_MAP_READ, 0, 0, 0);
    LARGE_INTEGER file_size;
    GetFileSizeEx(files.hFile, &file_size);
    files.data_size = file_size.QuadPart;
    #else
    files.fd = open(INPUT_FILE, O_RDONLY);
    struct stat sb;
    fstat(files.fd, &sb);
    files.data_size = sb.st_size;
    files.file_data = mmap(NULL, files.data_size, PROT_READ, MAP_PRIVATE, files.fd, 0);
    madvise(files.file_data, files.data_size, MADV_SEQUENTIAL);
    #endif

    #if MODE_WRITE == 1
    #ifdef _WIN32
    HANDLE hFileOut = CreateFileA(OUTPUT_FILE, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    files.writer.handle = hFileOut;
    #else
    files.writer.fd = open(OUTPUT_FILE, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    #endif
    const char* header = WRITE_CSV_HEADER;
    size_t header_len = strlen(header);
    memcpy(files.writer.buffer, header, header_len);
    files.writer.pos = header_len;
    #endif
    return files;
}

INLINE void free_files(Files* files) {
    #ifdef _WIN32
    UnmapViewOfFile(files->file_data);
    CloseHandle(files->hMapping);
    CloseHandle(files->hFile);
    #else
    munmap(files->file_data, files->data_size);
    close(files->fd);
    #endif

    #if MODE_WRITE == 1
    flush_buffer(&files->writer);
    #ifdef _WIN32
    CloseHandle(files->writer.handle);
    #else
    close(files->writer.fd);
    #endif
    #endif
}

#endif