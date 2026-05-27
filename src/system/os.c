#include "system/os.h"

#include <args.h>
#include <errno.h>
#include <print.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>

#ifdef _WIN32
#include <direct.h>
#include <malloc.h>
#include <windef.h>
#include <winbase.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define mkdir(dir, mode) _mkdir(dir)
#else
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>
#ifndef PATH_MAX
#define PATH_MAX _POSIX_PATH_MAX
#endif
#define MAX_PATH PATH_MAX
#endif

struct mmap {
#ifdef _WIN32
	HANDLE file, map;
#else
	size_t bytes;
	int fd;
#endif
};

void *alloc_mmap(size_t bytes)
{
	size_t total = sizeof(struct mmap) + bytes;
#ifdef _WIN32
	char dir[MAX_PATH] = {};
	char name[MAX_PATH] = "temporary matrix file";

	DWORD dir_len = GetTempPathA(MAX_PATH, dir);
	if (!dir_len || dir_len >= MAX_PATH) {
		perr("Could not resolve temp directory for '%s'",
		     file_name(name));
		return nullptr;
	}

	if (!GetTempFileNameA(dir, "sqa", 0, name)) {
		perr("Could not create temp file name for '%s'",
		     file_name(name));
		return nullptr;
	}

	HANDLE file = CreateFileA(
		name, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
	if (file == INVALID_HANDLE_VALUE) {
		perr("Could not create memory-mapped file '%s'",
		     file_name(name));
		return nullptr;
	}

	LARGE_INTEGER file_size;
	file_size.QuadPart = (LONGLONG)total;
	SetFilePointerEx(file, file_size, NULL, FILE_BEGIN);
	SetEndOfFile(file);

	HANDLE map = CreateFileMappingA(file, NULL, PAGE_READWRITE, 0, 0, NULL);
	if (!map) {
		perr("Could not create file mapping for '%s'", file_name(name));
		CloseHandle(file);
		return nullptr;
	}

	struct mmap *m = MapViewOfFile(map, FILE_MAP_ALL_ACCESS, 0, 0, 0);
	if (!m) {
		perr("Could not map view of file '%s'", file_name(name));
		CloseHandle(map);
		CloseHandle(file);
		return nullptr;
	}
	m->file = file;
	m->map = map;
#else
	char name[] = "/tmp/seqalign-mmap-XXXXXX";
	int fd = mkstemp(name);
	if (fd == -1) {
		perr("Could not create memory-mapped file '%s'",
		     file_name(name));
		return nullptr;
	}

	if (unlink(name) == -1) {
		perr("Could not unlink memory-mapped file '%s'",
		     file_name(name));
		close(fd);
		return nullptr;
	}

	if (ftruncate(fd, (off_t)total) == -1) {
		perr("Could not set size for file '%s'", file_name(name));
		close(fd);
		return nullptr;
	}

	struct mmap *m =
		mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (m == MAP_FAILED) {
		perr("Could not memory map file '%s'", file_name(name));
		close(fd);
		return nullptr;
	}

	m->fd = fd;
	m->bytes = bytes;
	madvise(m, total, MADV_RANDOM);
	madvise(m, total, MADV_HUGEPAGE);
	madvise(m, total, MADV_DONTFORK);
	madvise(m, total, MADV_DONTDUMP);
#endif
	return m + 1;
}

void free_mmap(void *mmap)
{
	if (!mmap)
		return;
	struct mmap *m = (struct mmap *)mmap - 1;
#ifdef _WIN32
	UnmapViewOfFile(m);
	CloseHandle(m->map);
	CloseHandle(m->file);
#else
	munmap(m, sizeof(*m) + m->bytes);
	close(m->fd);
#endif
}

#ifdef _WIN32
static double FREQ_INV;

[[gnu::constructor]]
static void time_init(void)
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	FREQ_INV = 1.0 / (double)freq.QuadPart;
}

double time_current(void)
{
	LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	return (double)count.QuadPart * FREQ_INV;
}

#else

double time_current(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif

size_t available_memory(void)
{
	size_t available_mem = 0;
#ifdef _WIN32
	MEMORYSTATUSEX status;
	status.dwLength = sizeof(status);
	GlobalMemoryStatusEx(&status);
	available_mem = status.ullAvailPhys;
#else
	FILE *fp = fopen("/proc/meminfo", "r");
	if unlikely (!fp)
		goto file_error;

	char line[256];
	while (fgets(line, sizeof(line), fp)) {
		if (strncmp(line, "MemAvailable:", 13) == 0) {
			char *endptr;
			auto val = strtoull(line + 13, &endptr, 10);
			if (endptr != line + 13) {
				available_mem = val * KiB;
				break;
			}
		}
	}
	fclose(fp);
file_error:
	if (!available_mem) {
		struct sysinfo info;
		if (sysinfo(&info) == 0)
			available_mem = info.freeram * info.mem_unit;
	}
#endif
	return available_mem;
}

void free_aligned(void *ptr)
{
#ifdef _WIN32
	_aligned_free(ptr);
#else
	free(ptr);
#endif
}

void *alloc_aligned(size_t alignment, size_t bytes)
{
	if unlikely (alignment < sizeof(void *) || alignment & (alignment - 1))
		return nullptr;

	if (bytes % alignment != 0)
		bytes = (bytes + alignment - 1) & ~(alignment - 1);

	return aligned_alloc(alignment, bytes);
}

const char *file_name(const char *path)
{
	if unlikely (!*path)
		return nullptr;
#ifdef _WIN32
	const char *name = strrchr(path, '\\');
#else
	const char *name = strrchr(path, '/');
#endif
	return name ? name + 1 : path;
}

bool path_special_exists(const char *path)
{
	if unlikely (!*path)
		return false;
#ifdef _WIN32
	DWORD attr = GetFileAttributesA(path);
	if (attr == INVALID_FILE_ATTRIBUTES)
		return false;

	return (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
	struct stat st;
	if (lstat(path, &st) != 0)
		return false;

	return (S_ISDIR(st.st_mode) || S_ISCHR(st.st_mode) ||
		S_ISBLK(st.st_mode) || S_ISFIFO(st.st_mode) ||
		S_ISSOCK(st.st_mode));
#endif
}

bool path_file_exists(const char *path)
{
	if unlikely (!*path)
		return false;
#ifdef _WIN32
	DWORD attr = GetFileAttributesA(path);
	if (attr == INVALID_FILE_ATTRIBUTES)
		return false;

	return true;
#else
	struct stat st;
	if (lstat(path, &st) != 0)
		return false;

	return (S_ISREG(st.st_mode) || S_ISLNK(st.st_mode));
#endif
}

[[gnu::nonnull]]
static const char *_find_last_sep(const char *path)
{
	const char *last1 = strrchr(path, '/');
#ifdef _WIN32
	const char *last2 = strrchr(path, '\\');
	if (!last1)
		return last2;

	if (!last2)
		return last1;

	return max(last1, last2);
#else
	return last1;
#endif
}

bool path_directories_create(const char *path)
{
	if unlikely (!*path)
		return false;

	const char *last_sep = _find_last_sep(path);
	if (!last_sep)
		return true;

	size_t dir_len = (size_t)(last_sep - path);
	if (dir_len == 0)
		return true;

	char *MALLOCA(dirbuf, dir_len + 1);
	if unlikely (!dirbuf)
		return false;

	memcpy(dirbuf, path, dir_len);
	dirbuf[dir_len] = '\0';

	char *p = dirbuf;
	if (*p == '/' || *p == '\\')
		p++;

	for (; *p; ++p) {
		if (*p != '/' && *p != '\\')
			continue;

		char saved = *p;
		*p = '\0';

		if (mkdir(dirbuf, 0755) != 0 && errno != EEXIST) {
			free(dirbuf);
			return false;
		}

		*p = saved;
	}

	if (mkdir(dirbuf, 0755) != 0 && errno != EEXIST) {
		free(dirbuf);
		return false;
	}

	free(dirbuf);
	return true;
}

struct arg_callback parse_path(const char *str, void *dest)
{
	if (strlen(str) >= MAX_PATH)
		return ARG_INVALID("File path is too long");

	if (path_special_exists(str))
		return ARG_INVALID("Path is a directory or non-regular file");

	*(const char **)dest = str;
	return ARG_VALID();
}

int THREAD_NUM;

ARG_PARSE_UL(thread_num, 10, int, (int), val > INT_MAX, "Invalid thread count")

static struct arg_callback validate_thread_num(void)
{
	if (THREAD_NUM)
		omp_set_num_threads(THREAD_NUM);
	else
		THREAD_NUM = omp_get_max_threads();
	return ARG_VALID();
}

static void print_threads(void)
{
	pinfol("CPU Threads: %d", THREAD_NUM);
}

ARG_EXTERN(disable_cuda);
ARG_EXTERN(benchmark);

ARGUMENT(threads) = {
	.opt = 'T',
	.lopt = "threads",
	.help = "Number of threads (0 = auto)",
	.param = "N",
	.param_req = ARG_PARAM_REQUIRED,
	.dest = &THREAD_NUM,
	.parse_callback = parse_thread_num,
	.validate_callback = validate_thread_num,
	.action_callback = print_threads,
	.action_order = ARG_ORDER_AFTER(ARG(disable_cuda)),
	.help_order = ARG_ORDER_AFTER(ARG(benchmark)),
};
