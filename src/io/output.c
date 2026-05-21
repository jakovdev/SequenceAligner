#include "io/output.h"

#include <args.h>
#include <print.h>

#include "bio/sequences.h"
#include "interface/seqalign_cuda.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/benchmark.h"

static bool disable_write;
static const char *OUTPUT_PATH;

[[gnu::nonnull]]
static void mmap_init(struct mmap *file)
{
#ifdef _WIN32
	file->hFile = INVALID_HANDLE_VALUE;
	file->hMapping = nullptr;
#else
	file->fd = -1;
#endif
}

[[gnu::nonnull]]
static void mmap_close(struct mmap *file)
{
#ifdef _WIN32
	if (file->hMapping) {
		CloseHandle(file->hMapping);
		file->hMapping = nullptr;
	}

	if (file->hFile != INVALID_HANDLE_VALUE) {
		CloseHandle(file->hFile);
		file->hFile = INVALID_HANDLE_VALUE;
	}
#else
	if (file->fd != -1) {
		close(file->fd);
		file->fd = -1;
	}
#endif
}

[[gnu::nonnull]]
static bool mmap_open(struct output *sm)
{
	mmap_init(&sm->file);

#define file_error_return(message_lit)                      \
	do {                                                \
		perr(message_lit " '%s'", file_name(name)); \
		mmap_close(&sm->file);                      \
		return false;                               \
	} while (0)

#ifdef _WIN32
	char dir[MAX_PATH] = {};
	char name[MAX_PATH] = "temporary matrix file";

	DWORD dir_len = GetTempPathA(MAX_PATH, dir);
	if (!dir_len || dir_len >= MAX_PATH)
		file_error_return("Could not resolve temp directory for");

	if (!GetTempFileNameA(dir, "sqa", 0, name))
		file_error_return("Could not create temp file name for");

	sm->file.hFile = CreateFileA(
		name, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
		FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
	if (sm->file.hFile == INVALID_HANDLE_VALUE)
		file_error_return("Could not create memory-mapped file");

	LARGE_INTEGER file_size;
	file_size.QuadPart = (LONGLONG)sm->bytes;
	SetFilePointerEx(sm->file.hFile, file_size, nullptr, FILE_BEGIN);
	SetEndOfFile(sm->file.hFile);

	sm->file.hMapping = CreateFileMapping(sm->file.hFile, nullptr,
					      PAGE_READWRITE, 0, 0, nullptr);
	if (!sm->file.hMapping)
		file_error_return("Could not create file mapping for");

	sm->matrix =
		MapViewOfFile(sm->file.hMapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
	if (!sm->matrix)
		file_error_return("Could not map view of file");
#else
	char name[] = "/tmp/seqalign-mmap-XXXXXX";
	sm->file.fd = mkstemp(name);
	if (sm->file.fd == -1)
		file_error_return("Could not create memory-mapped file");

	if (unlink(name) == -1)
		file_error_return("Could not unlink memory-mapped file");

	if (ftruncate(sm->file.fd, (off_t)sm->bytes) == -1)
		file_error_return("Could not set size for file");

	sm->matrix = mmap(nullptr, sm->bytes, PROT_READ | PROT_WRITE,
			  MAP_SHARED, sm->file.fd, 0);
	if (sm->matrix == MAP_FAILED)
		file_error_return("Could not memory map file");

	madvise(sm->matrix, sm->bytes, MADV_RANDOM);
	madvise(sm->matrix, sm->bytes, MADV_HUGEPAGE);
	madvise(sm->matrix, sm->bytes, MADV_DONTFORK);
	madvise(sm->matrix, sm->bytes, MADV_DONTDUMP);
#endif
	return true;

#undef file_error_return
}

bool output_load(struct output *sm, const struct input *in)
{
	if (disable_write)
		return true;

	sm->dim = (size_t)in->seqs_n;
	MALLOCA(sm->seqs, sm->dim);
	if (!sm->seqs) {
		perr("Out of memory allocating output sequence data");
		return false;
	}

	for (size_t i = 0; i < sm->dim; i++)
		sm->seqs[i] = in->seqs[i].letters;

	const size_t safe = available_memory() * 3 / 4;
	sm->bytes = bytesof(sm->matrix, sm->dim * sm->dim);
	if (!cuda_memory(sm->bytes) || sm->bytes > safe) {
		sm->bytes = bytesof(sm->matrix, sm->dim * (sm->dim - 1) / 2);
		sm->mmap = sm->bytes > safe;
		sm->triangular = true;
		pverb("Using triangular matrix storage");
	}

	bench_output_start();
	if (sm->mmap) {
		pinfo("Similarity Matrix size exceeds memory limits");
		pinfol("Creating temporary matrix file (%.2f GiB)",
		       (double)sm->bytes / (double)GiB);
		if (!mmap_open(sm))
			return false;
	} else {
		MALLOC_AL(sm->matrix, PAGE_SIZE, sm->bytes);
		if unlikely (!sm->matrix) {
			perr("Out of memory allocating Similarity Matrix");
			return false;
		}
		memset(sm->matrix, 0, sm->bytes);
	}
	bench_output_end();

	pinfo("Similarity Matrix size: %zu x %zu", sm->dim, sm->dim);
	return true;
}

void output_fill(struct output *sm, s32 col, const s32 *columns)
{
	if (disable_write)
		return;

	if (!sm->matrix || col < 0 || (size_t)col >= sm->dim)
		unreachable_release();

	if (sm->triangular) {
		memcpy(sm->matrix + ((s64)col * (col - 1)) / 2, columns,
		       bytesof(sm->matrix, (size_t)col));
	} else {
		for (s32 row = 0; row < col; row++) {
			sm->matrix[sm->dim * row + col] = columns[row];
			sm->matrix[sm->dim * col + row] = columns[row];
		}
	}
}

flush_fn FLUSH_FORMATS[FLUSH_COUNT];
enum output_format FLUSH_ID = FLUSH_HDF5 /* FLUSH_INVALID */;

bool output_flush(struct output *sm)
{
	if (disable_write)
		return true;
	bench_output_start();
	bool retval = FLUSH_FORMATS[FLUSH_ID](sm, OUTPUT_PATH);
	bench_output_end();
	bench_output_print();
	return retval;
}

void output_free(struct output *sm)
{
	free(sm->seqs);
	if (!sm->mmap) {
		free_aligned(sm->matrix);
	} else {
#ifdef _WIN32
		if (sm->matrix)
			UnmapViewOfFile(sm->matrix);
#else
		if (sm->matrix && sm->matrix != MAP_FAILED)
			munmap(sm->matrix, sm->bytes);
#endif
		mmap_close(&sm->file);
	}
	memset(sm, 0, sizeof(*sm));
}

ARG_EXTERN(disable_cuda);

ARGUMENT(disable_write) = {
	.opt = 'W',
	.lopt = "no-write",
	.help = "Disable writing to output file",
	.set = &disable_write,
	.help_order = ARG_ORDER_AFTER(ARG(disable_cuda)),
};

static void print_output_path(void)
{
	if (disable_write)
		pwarnm("Output: Ignored");
	else
		pinfom("Output: %s", file_name(OUTPUT_PATH));
}

static struct arg_callback validate_output_path(void)
{
	if (disable_write)
		return ARG_VALID();

	if (path_file_exists(OUTPUT_PATH)) {
		pwarn("Output file already exists: %s", file_name(OUTPUT_PATH));
		if (!print_yN("Do you want to DELETE it?"))
			return ARG_INVALID(
				"Output file exists and will not be overwritten");
		if (remove(OUTPUT_PATH) != 0)
			return ARG_INVALID(
				"Failed to delete existing output file");
		pinfo("Deleted existing output file");
	}

	if (!path_directories_create(OUTPUT_PATH))
		return ARG_INVALID(
			"Failed to create directories for output file");

	return ARG_VALID();
}

ARG_EXTERN(input_path);

ARGUMENT(output_path) = {
	.opt = 'o',
	.lopt = "output",
	.help = "Output file path: HDF5 format",
	.param = "FILE",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = &OUTPUT_PATH,
	.parse_callback = parse_path,
	.validate_callback = validate_output_path,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_AFTER(ARG(input_path)),
	.action_callback = print_output_path,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(input_path)),
	.help_order = ARG_ORDER_AFTER(ARG(input_path)),
	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(disable_write)),
};
