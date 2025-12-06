#include "io/files.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bio/types.h"
#include "io/format/csv.h"
#include "io/format/fasta.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/args.h"
#include "util/print.h"

static void file_metadata_init(struct FileMetadata *meta)
{
#ifdef _WIN32
	meta->hFile = INVALID_HANDLE_VALUE;
	meta->hMapping = NULL;
#else
	meta->fd = -1;
#endif
	meta->bytes = 0;
}

static void file_metadata_close(struct FileMetadata *meta)
{
#ifdef _WIN32
	if (meta->hMapping) {
		CloseHandle(meta->hMapping);
		meta->hMapping = NULL;
	}

	if (meta->hFile != INVALID_HANDLE_VALUE) {
		CloseHandle(meta->hFile);
		meta->hFile = INVALID_HANDLE_VALUE;
	}

#else
	if (meta->fd != -1) {
		close(meta->fd);
		meta->fd = -1;
	}

#endif
	meta->bytes = 0;
}

static enum FileFormat file_format_detect(const char *file_path)
{
	const char *ext = strrchr(file_path, '.');
	if (!ext)
		return FILE_FORMAT_UNKNOWN;

	ext++;

	if (strcasecmp(ext, "csv") == 0)
		return FILE_FORMAT_CSV;
	else if (strcasecmp(ext, "fasta") == 0 || strcasecmp(ext, "fa") == 0 ||
		 strcasecmp(ext, "fas") == 0 || strcasecmp(ext, "fna") == 0 ||
		 strcasecmp(ext, "ffn") == 0 || strcasecmp(ext, "faa") == 0 ||
		 strcasecmp(ext, "frn") == 0 || strcasecmp(ext, "mpfa") == 0)
		return FILE_FORMAT_FASTA;

	return FILE_FORMAT_UNKNOWN;
}

static void file_format_data_reset(struct FileFormatMetadata *data)
{
	memset(data, 0, sizeof(*data));
}

static bool file_format_csv_parse(struct FileText *file)
{
	if (!file->text)
		return false;

	perr_context("CSV");

	if (!csv_validate(file->data.start, file->data.end))
		return false;

	char *file_header_start =
		csv_header_parse(file->data.start, file->data.end,
				 &file->data.format.csv.headerless,
				 &file->data.format.csv.sequence_column);

	file->data.start = file->data.format.csv.headerless ? file->text :
							      file_header_start;
	file->data.cursor = file->data.start;

	pverb("Counting sequences in input file");
	size_t total = csv_total_lines(file->data.start, file->data.end);

	if (total >= SEQUENCE_COUNT_MAX) {
		perr("Too many sequences in input file: %zu", total);
		return false;
	}

	if (!total) {
		perr("No sequences found in input file");
		return false;
	}

	file->data.total = (s32)total;
	pinfo("Found " Ps32 " sequences", file->data.total);
	return true;
}

static bool file_format_fasta_parse(struct FileText *file)
{
	if (!file->text)
		return false;

	perr_context("FASTA");

	if (!fasta_validate(file->data.start, file->data.end))
		return false;

	pverb("Counting sequences in input file");
	size_t total = fasta_total_entries(file->data.start, file->data.end);

	if (total >= SEQUENCE_COUNT_MAX) {
		perr("Too many sequences in input file: %zu", total);
		return false;
	}

	if (!total) {
		perr("No sequences found in input file");
		return false;
	}

	file->data.total = (s32)total;
	pinfo("Found " Ps32 " sequences", file->data.total);
	return true;
}

void file_text_close(struct FileText *file)
{
#ifdef _WIN32
	if (file->text) {
		UnmapViewOfFile(file->text);
		file->text = NULL;
	}

#else
	if (file->text) {
		munmap(file->text, file->meta.bytes);
		file->text = NULL;
	}

#endif
	file_metadata_close(&file->meta);
	file_format_data_reset(&file->data);
}

bool file_text_open(struct FileText *restrict file, const char *restrict path)
{
	perr_context("FILE");

	file_metadata_init(&file->meta);
	file_format_data_reset(&file->data);
	file->text = NULL;

	const char *file_name = file_name_path(path);
#define file_error_defer(message_lit)                 \
	do {                                          \
		perr(message_lit " '%s'", file_name); \
		goto file_error;                      \
	} while (0)

#ifdef _WIN32
	file->meta.hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
				       NULL, OPEN_EXISTING,
				       FILE_FLAG_SEQUENTIAL_SCAN, NULL);

	if (file->meta.hFile == INVALID_HANDLE_VALUE) {
		perr("Could not open file '%s'", file_name);
		return false;
	}

	file->meta.hMapping = CreateFileMapping(file->meta.hFile, NULL,
						PAGE_READONLY, 0, 0, NULL);
	if (!file->meta.hMapping)
		file_error_defer("Could not create file mapping for");

	file->text = MapViewOfFile(file->meta.hMapping, FILE_MAP_READ, 0, 0, 0);
	if (!file->text)
		file_error_defer("Could not map view of file '%s'");

	LARGE_INTEGER file_size;
	if (!GetFileSizeEx(file->meta.hFile, &file_size))
		file_error_defer("Could not get file size for '%s'");

	file->meta.bytes = (size_t)file_size.QuadPart;
#else
	file->meta.fd = open(path, O_RDONLY);
	if (file->meta.fd == -1)
		file_error_defer("Could not open input file '%s'");

	struct stat sb;
	if (fstat(file->meta.fd, &sb) == -1)
		file_error_defer("Could not stat file '%s'");

	if (!S_ISREG(sb.st_mode) || sb.st_size < 0)
		file_error_defer("Invalid file type or size for '%s'");

	file->meta.bytes = (size_t)sb.st_size;
	file->text = mmap(NULL, file->meta.bytes, PROT_READ, MAP_PRIVATE,
			  file->meta.fd, 0);
	if (file->text == MAP_FAILED)
		file_error_defer("Could not memory map file '%s'");

	madvise(file->text, file->meta.bytes, MADV_SEQUENTIAL);
#endif
#undef file_error_defer

	file->data.start = file->text;
	file->data.end = file->text + file->meta.bytes;
	file->data.cursor = file->data.start;

	enum FileFormat type = file_format_detect(path);
	file->data.type = type;

	switch (type) {
	case FILE_FORMAT_CSV:
		return file_format_csv_parse(file);
	case FILE_FORMAT_FASTA:
		return file_format_fasta_parse(file);
	case FILE_FORMAT_UNKNOWN:
	default:
		perr("Failed to parse file format");
		break;
	}

file_error:
	file_text_close(file);
	return false;
}

s32 file_sequence_total(struct FileText *file)
{
	if (file)
		return file->data.total;

	perr_context("FILE");
	perr("Invalid file for total sequence count");
	exit(EXIT_FAILURE);
}

size_t file_sequence_next_length(struct FileText *file)
{
	if (file) {
		switch (file->data.type) {
		case FILE_FORMAT_CSV:
			return csv_line_column_length(
				file->data.cursor,
				file->data.format.csv.sequence_column);
		case FILE_FORMAT_FASTA:
			return fasta_entry_length(file->data.cursor,
						  file->data.end);
		case FILE_FORMAT_UNKNOWN:
		default:
			break;
		}
	}

	perr_context("FILE");
	perr("Invalid file for sequence column length");
	exit(EXIT_FAILURE);
}

bool file_sequence_next(struct FileText *file)
{
	if (file) {
		switch (file->data.type) {
		case FILE_FORMAT_CSV:
			return csv_line_next(&file->data.cursor);
		case FILE_FORMAT_FASTA:
			return fasta_entry_next(&file->data.cursor);
		case FILE_FORMAT_UNKNOWN:
		default:
			break;
		}
	}

	perr_context("FILE");
	perr("Invalid file for next sequence line");
	exit(EXIT_FAILURE);
}

size_t file_extract_entry(struct FileText *restrict file, char *restrict out)
{
	if (file && out) {
		switch (file->data.type) {
		case FILE_FORMAT_CSV:
			return csv_line_column_extract(
				&file->data.cursor, out,
				file->data.format.csv.sequence_column);
		case FILE_FORMAT_FASTA:
			return fasta_entry_extract(&file->data.cursor,
						   file->data.end, out);
		case FILE_FORMAT_UNKNOWN:
		default:
			break;
		}
	}

	perr_context("FILE");
	perr("Invalid file for sequence extraction");
	exit(EXIT_FAILURE);
}

bool file_matrix_open(struct FileScoreMatrix *restrict file,
		      const char *restrict path, size_t matrix_dim)
{
	file_metadata_init(&file->meta);

	size_t triangle_elements = (matrix_dim * (matrix_dim - 1)) / 2;
	size_t bytes = triangle_elements * sizeof(*file->matrix);
	file->meta.bytes = bytes;
	const char *file_name = file_name_path(path);
#define file_error_return(message_lit)                \
	do {                                          \
		perr(message_lit " '%s'", file_name); \
		file_matrix_close(file);              \
		return false;                         \
	} while (0)

	const double mmap_size = (double)bytes / (double)GiB;
	pinfol("Creating matrix file: %s (%.2f GiB)", file_name, mmap_size);
	perr_context("MATRIXFILE");

#ifdef _WIN32
	file->meta.hFile = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0,
				       NULL, CREATE_ALWAYS,
				       FILE_FLAG_DELETE_ON_CLOSE, NULL);

	if (file->meta.hFile == INVALID_HANDLE_VALUE)
		file_error_return("Could not create memory-mapped file '%s'");

	LARGE_INTEGER file_size;
	file_size.QuadPart = (LONGLONG)bytes;
	SetFilePointerEx(file->meta.hFile, file_size, NULL, FILE_BEGIN);
	SetEndOfFile(file->meta.hFile);

	file->meta.hMapping = CreateFileMapping(file->meta.hFile, NULL,
						PAGE_READWRITE, 0, 0, NULL);
	if (!file->meta.hMapping)
		file_error_return("Could not create file mapping for '%s'");

	file->matrix = MapViewOfFile(file->meta.hMapping, FILE_MAP_ALL_ACCESS,
				     0, 0, 0);
	if (!file->matrix)
		file_error_return("Could not map view of file '%s'");

#else
	file->meta.fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
	if (file->meta.fd == -1)
		file_error_return("Could not create memory-mapped file '%s'");

	if (unlink(path) == -1)
		file_error_return("Could not unlink memory-mapped file '%s'");

	if (ftruncate(file->meta.fd, (off_t)bytes) == -1)
		file_error_return("Could not set size for file '%s'");

	file->matrix = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
			    file->meta.fd, 0);
	if (file->matrix == MAP_FAILED)
		file_error_return("Could not memory map file '%s'");

	madvise(file->matrix, bytes, MADV_RANDOM);
	madvise(file->matrix, bytes, MADV_HUGEPAGE);
	madvise(file->matrix, bytes, MADV_DONTFORK);
	madvise(file->matrix, bytes, MADV_DONTDUMP);

#endif

	const size_t check_indices[5] = { 0, triangle_elements / 4,
					  triangle_elements / 2,
					  triangle_elements * 3 / 4,
					  triangle_elements - 1 };

	bool is_zeroed = true;
	for (size_t i = 0; i < ARRAY_SIZE(check_indices); i++) {
		if (file->matrix[check_indices[i]] != 0) {
			is_zeroed = false;
			break;
		}
	}

	if (!is_zeroed)
		memset(file->matrix, 0, bytes);

	return true;
}

void file_matrix_close(struct FileScoreMatrix *file)
{
	if (!file)
		return;

	file_metadata_close(&file->meta);
	if (!file->matrix)
		return;

#ifdef _WIN32
	UnmapViewOfFile(file->matrix);
#else
	munmap(file->matrix, file->meta.bytes);
#endif
	file->matrix = NULL;
}

s64 matrix_index(s32 row, s32 col)
{
	return ((s64)col * (col - 1)) / 2 + row;
}

void file_matrix_name(char *buffer, size_t buffer_size, const char *output_path)
{
	if (!output_path || !output_path[0]) {
		snprintf(buffer, buffer_size, "./seqalign_matrix.mmap");
		return;
	}

	char dir[MAX_PATH] = { 0 };
	char base[MAX_PATH] = { 0 };

	const char *last_slash = strrchr(output_path, '/');
	if (last_slash) {
		ptrdiff_t delta = last_slash - output_path + 1;
		size_t dir_len = delta < 0 ? 0 : (size_t)delta;
		strncpy(dir, output_path, dir_len);
		dir[dir_len] = '\0';
		strncpy(base, last_slash + 1, MAX_PATH - 1);
	} else {
		strcpy(dir, "./");
		strncpy(base, output_path, MAX_PATH - 1);
	}

	base[MAX_PATH - 1] = '\0';
	char *dot = strrchr(base, '.');
	if (dot)
		*dot = '\0';

	snprintf(buffer, buffer_size, "%s%s.mmap", dir, base);
}

static bool no_write;

ARGUMENT(disable_write) = {
	.opt = 'W',
	.lopt = "no-write",
	.help = "Disable writing to output file",
	.set = &no_write,
	.help_weight = 400,
};

bool arg_mode_write(void)
{
	return !no_write;
}

static char input_path[MAX_PATH];
static char output_path[MAX_PATH];

const char *arg_input(void)
{
	return input_path;
}

const char *arg_output(void)
{
	return output_path;
}

static void print_input_path(void)
{
	pinfo("Input: %s", file_name_path(input_path));
}

static void print_output_path(void)
{
	if (no_write)
		pwarnm("Output: Ignored");
	else
		pinfom("Output: %s", file_name_path(output_path));
}

static struct arg_callback parse_path(const char *str, void *dest)
{
	if (strlen(str) >= MAX_PATH)
		return ARG_INVALID("File path is too long");

	if (path_special_exists(str))
		return ARG_INVALID("Path is a directory or non-regular file");

	snprintf(dest, MAX_PATH, "%s", str);
	return ARG_VALID();
}

static struct arg_callback validate_input_path(void)
{
	if (!path_file_exists(input_path))
		return ARG_INVALID("Input file does not exist");

	return ARG_VALID();
}

static struct arg_callback validate_output_path(void)
{
	if (no_write)
		return ARG_VALID();

	if (path_file_exists(output_path)) {
		pwarn("Output file already exists: %s",
		      file_name_path(output_path));
		if (!print_yN("Do you want to DELETE it?"))
			return ARG_INVALID(
				"Output file exists and will not be overwritten");
		if (remove(output_path) != 0)
			return ARG_INVALID(
				"Failed to delete existing output file");
		pinfo("Deleted existing output file");
	}

	if (!path_directories_create(output_path))
		return ARG_INVALID(
			"Failed to create directories for output file");

	return ARG_VALID();
}

ARGUMENT(input_path) = {
	.opt = 'i',
	.lopt = "input",
	.help = "Input file path (FASTA, CSV)",
	.param = "FILE",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = input_path,
	.parse_callback = parse_path,
	.validate_callback = validate_input_path,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_weight = 1000,
	.action_callback = print_input_path,
	.action_phase = ARG_CALLBACK_ALWAYS,
	.action_weight = 1000,
	.help_weight = 1000,
};

ARGUMENT(output_path) = {
	.opt = 'o',
	.lopt = "output",
	.help = "Output file path (HDF5)",
	.param = "FILE",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = output_path,
	.parse_callback = parse_path,
	.validate_callback = validate_output_path,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_weight = 900,
	.action_callback = print_output_path,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_weight = 900,
	.help_weight = 900,
	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(disable_write)),
};
