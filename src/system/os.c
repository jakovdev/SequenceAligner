#ifdef _WIN32
#include <direct.h>
#include <malloc.h>
#include <windef.h>
#include <winbase.h>
#else
#define _GNU_SOURCE
#include <fcntl.h>
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

#include <args.h>
#include <errno.h>
#include <print.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <omp.h>

#include "system/os.h"

void *alloc_mmap(size_t bytes)
{
#ifdef _WIN32
	if (bytes > LONG_LONG_MAX) {
		pdev("alloc_mmap bytes parameter exceeds maximum supported size");
		perr("Internal error creating temporary file");
		return nullptr;
	}

	char dir[MAX_PATH] = {};
	char name[MAX_PATH] = {};

	DWORD dir_len = GetTempPathA(MAX_PATH, dir);
	if (!dir_len || dir_len >= MAX_PATH) {
		perr("Could not retrieve the temporary directory path");
		return nullptr;
	}

	if (!GetTempFileNameA(dir, "sqa", 0, name)) {
		perr("Could not create a temporary file name");
		return nullptr;
	}

	HANDLE fd = CreateFileA(
		name, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
		FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
	if (fd == INVALID_HANDLE_VALUE) {
		perr("Could not create a temporary file");
		return nullptr;
	}

	LARGE_INTEGER sz;
	sz.QuadPart = (LONGLONG)bytes;
	if (!SetFilePointerEx(fd, sz, NULL, FILE_BEGIN) || !SetEndOfFile(fd)) {
		perr("Could not set temporary file size");
		CloseHandle(fd);
		return NULL;
	}

	HANDLE fm = CreateFileMappingA(fd, NULL, PAGE_READWRITE, 0, 0, NULL);
	CloseHandle(fd);
	if (!fm) {
		perr("Could not create temporary file mapping");
		return nullptr;
	}

	void *m = MapViewOfFile(fm, FILE_MAP_ALL_ACCESS, 0, 0, 0);
	CloseHandle(fm);
	if (!m) {
		perr("Could not memory map temporary file");
		return nullptr;
	}
	return m;
#else
	bytes += sizeof(size_t);
	if (bytes > LONG_MAX) {
		pdev("alloc_mmap bytes parameter exceeds maximum supported size");
		perr("Internal error creating temporary file");
		return nullptr;
	}

	int fd = open("/tmp", O_TMPFILE | O_RDWR, S_IRUSR | S_IWUSR);
	if (fd == -1) {
		perr("Could not create a temporary file");
		return nullptr;
	}

	if (ftruncate(fd, (off_t)bytes) == -1) {
		perr("Could not set size for temporary file");
		close(fd);
		return nullptr;
	}

	void *m = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (m == MAP_FAILED) {
		perr("Could not memory map temporary file");
		return nullptr;
	}

	madvise(m, bytes, MADV_RANDOM);
	madvise(m, bytes, MADV_HUGEPAGE);
	madvise(m, bytes, MADV_DONTFORK);
	madvise(m, bytes, MADV_DONTDUMP);
	*(size_t *)m = bytes;
	return m + sizeof(size_t);
#endif
}

void free_mmap(void *mmap)
{
	if (!mmap)
		return;
#ifdef _WIN32
	UnmapViewOfFile(mmap);
#else
	size_t *m = mmap - sizeof(size_t);
	munmap(m, *m);
#endif
}

bool read_mmap(const char *path, void **begin, void **end)
{
#ifdef _WIN32
	HANDLE fd = CreateFileA(
		path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (fd == INVALID_HANDLE_VALUE) {
		perr("Could not open file: %s", file_name(path));
		return false;
	}

	LARGE_INTEGER st;
	if (!GetFileSizeEx(fd, &st) || !st.QuadPart) {
		perr("Failed to get file size or empty: %s", file_name(path));
		CloseHandle(fd);
		return false;
	}

	HANDLE fmh = CreateFileMappingA(fd, NULL, PAGE_READONLY, 0, 0, NULL);
	CloseHandle(fd);
	if (!fmh) {
		perr("Failed to create file mapping: %s", file_name(path));
		return false;
	}

	void *fm = MapViewOfFile(fmh, FILE_MAP_READ, 0, 0, 0);
	CloseHandle(fmh);
	if (!fm) {
		perr("Failed to map file: %s", file_name(path));
		return false;
	}
	*end = fm + st.QuadPart;
#else
	int fd = open(path, O_RDONLY);
	if (fd < 0) {
		perr("Could not open file: %s", file_name(path));
		return false;
	}

	struct stat st;
	if (fstat(fd, &st) < 0) {
		perr("Failed to stat file size: %s", file_name(path));
		close(fd);
		return false;
	}

	if (st.st_size == 0) {
		perr("Empty file: %s", file_name(path));
		close(fd);
		return false;
	}

	void *fm = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (fm == MAP_FAILED) {
		perr("Failed to map file: %s", file_name(path));
		return false;
	}

	madvise(fm, st.st_size, MADV_SEQUENTIAL);
	madvise(fm, st.st_size, MADV_HUGEPAGE);
	madvise(fm, st.st_size, MADV_DONTFORK);
	madvise(fm, st.st_size, MADV_DONTDUMP);
	*end = fm + st.st_size;
#endif
	*begin = fm;
	return true;
}

void unread_mmap(void *begin, [[maybe_unused]] const void *end)
{
#ifdef _WIN32
	UnmapViewOfFile(begin);
#else
	munmap(begin, end - begin);
#endif
}

#ifdef _WIN32
static double FREQ_INV;

#if defined(__MINGW64__) && defined(USE_CUDA)
static void safe_exit(void)
{
	ExitProcess(0);
}
#endif

[[gnu::constructor]]
static void time_init(void)
{
#if defined(__MINGW64__) && defined(USE_CUDA)
	atexit(safe_exit);
#endif
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
	if (!fp)
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
#ifdef _WIN32
	return _aligned_malloc(bytes, alignment);
#else
	return aligned_alloc(alignment, bytes);
#endif
}

const char *file_name(const char *path)
{
	if (!*path)
		return nullptr;
	const char *name1 = strrchr(path, '/');
#ifdef _WIN32
	const char *name2 = strrchr(path, '\\');
	if (!name1)
		name1 = name2;
	else if (name2)
		name1 = max(name1, name2);
#endif
	return name1 ? name1 + 1 : path;
}

bool path_special_exists(const char *path)
{
	if (!*path)
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
	if (!*path)
		return false;
#ifdef _WIN32
	return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
	struct stat st;
	if (lstat(path, &st) != 0)
		return false;

	return (S_ISREG(st.st_mode) || S_ISLNK(st.st_mode));
#endif
}

[[gnu::nonnull]]
static const char *find_last_sep(const char *path)
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
	if (!*path)
		return false;

	const char *last_sep = find_last_sep(path);
	if (!last_sep)
		return true;

	size_t dir_len = (size_t)(last_sep - path);
	if (dir_len == 0)
		return true;

	char *MALLOCA(dirbuf, dir_len + 1);
	if (!dirbuf)
		return false;

	memcpy(dirbuf, path, dir_len);
	dirbuf[dir_len] = '\0';

	char *p = dirbuf;
#ifdef _WIN32
	if (*p == '/' || *p == '\\')
		p++;
#else
	if (*p == '/')
		p++;
#endif
	int retval;
	for (; *p; ++p) {
#ifdef _WIN32
		if (*p != '/' && *p != '\\')
			continue;
#else
		if (*p != '/')
			continue;
#endif
		char saved = *p;
		*p = '\0';

#ifdef _WIN32
		retval = _mkdir(dirbuf);
#else
		retval = mkdir(dirbuf, 0755);
#endif
		if (retval < 0 && errno != EEXIST) {
			free(dirbuf);
			return false;
		}

		*p = saved;
	}

#ifdef _WIN32
	retval = _mkdir(dirbuf);
#else
	retval = mkdir(dirbuf, 0755);
#endif
	if (retval < 0 && errno != EEXIST) {
		free(dirbuf);
		return false;
	}

	free(dirbuf);
	return true;
}

struct arg_callback parse_path(const char *str, void *dest)
{
	if (strnlen(str, MAX_PATH + 1) > MAX_PATH)
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
