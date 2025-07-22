#pragma once
#ifndef FILES_H
#define FILES_H

#include "arch.h"
#include "biotypes.h"
#include "csv.h"
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

typedef enum
{
    FILE_FORMAT_CSV,
    FILE_FORMAT_FASTA,
    FILE_FORMAT_UNKNOWN
} FileFormat;

typedef struct
{
    sequence_count_t total;
    FileFormat type;
    char* start;
    char* end;

    union
    {
        struct
        {
            size_t sequence_column;
            bool headerless;
        } csv;

        struct
        {
        } fasta;
    } format;
} FileFormatMetadata;

typedef struct FileText
{
    FileMetadata meta;
    FileFormatMetadata data;
    char* text;
} FileText;

typedef struct
{
    FileMetadata meta;
    score_t* matrix;
} FileScoreMatrix;

typedef struct FileText* restrict FileTextPtr;

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

static inline FileFormat
file_format_detect(const char* file_path)
{
    const char* ext = strrchr(file_path, '.');
    if (!ext)
    {
        return FILE_FORMAT_UNKNOWN;
    }

    ext++; // Skip the dot

    if (strcasecmp(ext, "csv") == 0)
    {
        return FILE_FORMAT_CSV;
    }

    else if (strcasecmp(ext, "fasta") == 0 || strcasecmp(ext, "fa") == 0 ||
             strcasecmp(ext, "fas") == 0)
    {
        return FILE_FORMAT_FASTA;
    }

    return FILE_FORMAT_UNKNOWN;
}

static inline void
file_format_data_reset(FileFormatMetadata* data)
{
    memset(data, 0, sizeof(*data));
}

static inline bool
file_format_csv_parse(FileText* file)
{
    if (!file->text)
    {
        return false;
    }

    char* file_header_start = csv_header_parse(file->data.start,
                                               file->data.end,
                                               &file->data.format.csv.headerless,
                                               &file->data.format.csv.sequence_column);

    file->data.start = file->data.format.csv.headerless ? file->text : file_header_start;

    print(VERBOSE, MSG_LOC(LAST), "Counting sequences in input file");
    file->data.total = (sequence_count_t)csv_total_lines(file->data.start, file->data.end);

    if (file->data.total >= SEQUENCE_COUNT_MAX)
    {
        print(ERROR, MSG_NONE, "CSV | Too many lines in input file: %u", file->data.total);
        return false;
    }

    if (!file->data.total)
    {
        print(ERROR, MSG_NONE, "CSV | No sequences found in input file");
        return false;
    }

    print(DNA, MSG_NONE, "Found %u potential sequences", file->data.total);
    return true;
}

static inline bool
file_format_fasta_parse(FileText* file)
{
    // TODO
    print(ERROR, MSG_NONE, "FILE | FASTA format not yet supported");
    return false;
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
    file_format_data_reset(&file->data);
}

static inline bool
file_text_open(FileText* file, const char* file_path)
{
    file_metadata_init(&file->meta);
    file_format_data_reset(&file->data);
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
        return false;
    }

    file->meta.hMapping = CreateFileMapping(file->meta.hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (file->meta.hMapping == NULL)
    {
        print(ERROR, MSG_NONE, "FILE | Could not create file mapping for '%s'", file_name);
        file_metadata_close(&file->meta);
        return false;
    }

    file->text = (char*)MapViewOfFile(file->meta.hMapping, FILE_MAP_READ, 0, 0, 0);
    if (file->text == NULL)
    {
        print(ERROR, MSG_NONE, "FILE | Could not map view of file '%s'", file_name);
        file_metadata_close(&file->meta);
        return false;
    }

    LARGE_INTEGER file_size;
    GetFileSizeEx(file->meta.hFile, &file_size);
    file->meta.bytes = file_size.QuadPart;
#else
    file->meta.fd = open(file_path, O_RDONLY);
    if (file->meta.fd == -1)
    {
        print(ERROR, MSG_NONE, "FILE | Could not open input file '%s'", file_name);
        return false;
    }

    struct stat sb;
    if (fstat(file->meta.fd, &sb) == -1)
    {
        print(ERROR, MSG_NONE, "FILE | Could not stat file '%s'", file_name);
        file_metadata_close(&file->meta);
        return false;
    }

    if (!S_ISREG(sb.st_mode) || sb.st_size < 0)
    {
        print(ERROR, MSG_NONE, "FILE | Invalid file type or size for '%s'", file_name);
        file_metadata_close(&file->meta);
        return false;
    }

    file->meta.bytes = (size_t)sb.st_size;
    file->text = (char*)mmap(NULL, file->meta.bytes, PROT_READ, MAP_PRIVATE, file->meta.fd, 0);
    if (file->text == MAP_FAILED)
    {
        print(ERROR, MSG_NONE, "FILE | Could not memory map file '%s'", file_name);
        file_metadata_close(&file->meta);
        file->text = NULL;
        return false;
    }

    madvise(file->text, file->meta.bytes, MADV_SEQUENTIAL);
#endif

    file->data.start = file->text;
    file->data.end = file->text + file->meta.bytes;

    FileFormat type = file_format_detect(file_path);
    file->data.type = type;

    switch (type)
    {
        case FILE_FORMAT_CSV:
            return file_format_csv_parse(file);
        case FILE_FORMAT_FASTA:
            return file_format_fasta_parse(file);
        case FILE_FORMAT_UNKNOWN:
            print(ERROR, MSG_NONE, "FILE | Failed to parse file format");
            file_text_close(file);
            return false;
    }
}

static inline size_t
file_sequence_next_length(FileTextPtr file, char* cursor)
{
    if (!file || !cursor)
    {
        print(ERROR, MSG_NONE, "FILE | Invalid parameters for sequence column length");
        exit(1);
    }

    switch (file->data.type)
    {
        case FILE_FORMAT_CSV:
            return csv_line_column_length(cursor, file->data.format.csv.sequence_column);
        case FILE_FORMAT_FASTA:
            // TODO: Implement FASTA sequence length detection
            print(ERROR, MSG_NONE, "FASTA format not yet supported");
            exit(1);
        case FILE_FORMAT_UNKNOWN:
        default:
            print(ERROR, MSG_NONE, "Unknown file format");
            exit(1);
    }
}

static inline bool
file_sequence_next(FileTextPtr file, char* restrict* restrict p_cursor)
{
    if (!file || !p_cursor)
    {
        print(ERROR, MSG_NONE, "FILE | Invalid parameters for next sequence line");
        exit(1);
    }

    switch (file->data.type)
    {
        case FILE_FORMAT_CSV:
            return csv_line_next(p_cursor);
        case FILE_FORMAT_FASTA:
            // TODO: Implement FASTA line navigation
            print(ERROR, MSG_NONE, "FASTA format not yet supported");
            exit(1);
        case FILE_FORMAT_UNKNOWN:
        default:
            print(ERROR, MSG_NONE, "Unknown file format");
            exit(1);
    }
}

static inline size_t
file_extract_sequence(FileTextPtr file, char* restrict* restrict p_cursor, char* restrict output)
{
    if (!file || !p_cursor || !output)
    {
        print(ERROR, MSG_NONE, "FILE | Invalid parameters for sequence extraction");
        exit(1);
    }

    switch (file->data.type)
    {
        case FILE_FORMAT_CSV:
            return csv_line_column_extract(p_cursor, output, file->data.format.csv.sequence_column);
        case FILE_FORMAT_FASTA:
            // TODO
            print(ERROR, MSG_NONE, "FASTA format not yet supported");
            exit(1);
        case FILE_FORMAT_UNKNOWN:
        default:
            print(ERROR, MSG_NONE, "Unknown file format");
            exit(1);
    }
}

static inline FileScoreMatrix
file_matrix_open(const char* file_path, sequence_count_t matrix_dim)
{
    FileScoreMatrix file = { 0 };
    file_metadata_init(&file.meta);

    alignment_size_t triangle_elements = (matrix_dim * (matrix_dim - 1)) / 2;
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
        return file;
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
        return file;
    }

    file.matrix = (score_t*)MapViewOfFile(file.meta.hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    if (file.matrix == NULL)
    {
        print(ERROR, MSG_NONE, "MATRIXFILE | Could not map view of file '%s'", file_name);
        file_metadata_close(&file.meta);
        return file;
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

    file.matrix = (score_t*)mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, file.meta.fd, 0);
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

    const alignment_size_t check_indices[5] = {
        0,                         // First element
        triangle_elements / 4,     // First quarter
        triangle_elements / 2,     // Middle
        triangle_elements * 3 / 4, // Third quarter
        triangle_elements - 1      // Last element
    };

    const alignment_size_t num_check_indices = sizeof(check_indices) / sizeof(*check_indices);

    bool is_zeroed = true;
    for (alignment_size_t i = 0; i < num_check_indices; i++)
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
file_matrix_close(FileScoreMatrix* file)
{
    if (!file->matrix)
    {
        return;
    }

#ifdef _WIN32
    UnmapViewOfFile(file->matrix);
#else
    munmap(file->matrix, file->meta.bytes);
#endif
    file->matrix = NULL;
    file_metadata_close(&file->meta);
}

static inline alignment_size_t
matrix_triangle_index(sequence_index_t row, sequence_index_t col)
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
            ptrdiff_t delta = last_slash - output_path + 1;
            size_t dir_len = delta < 0 ? 0 : (size_t)delta;
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