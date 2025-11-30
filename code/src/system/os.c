#include "system/os.h"

#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include "util/args.h"
#include "util/print.h"

#ifdef _WIN32
#include <direct.h>
#define mkdir(dir, mode) _mkdir(dir)

static double g_freq_inv;

void time_init(void)
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	g_freq_inv = 1.0 / (double)freq.QuadPart;
}

double time_current(void)
{
	LARGE_INTEGER count;
	QueryPerformanceCounter(&count);
	return (double)count.QuadPart * g_freq_inv;
}

#else
#include <time.h>

double time_current(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif

const char *file_name_path(const char *path)
{
#ifdef _WIN32
	const char *name = strrchr(path, '\\');
#else
	const char *name = strrchr(path, '/');
#endif
	return name ? name + 1 : path;
}

bool path_special_exists(const char *path)
{
	if (!path || path[0] == '\0')
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
	if (!path || path[0] == '\0')
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

static const char *_find_last_sep(const char *path)
{
	const char *last1 = strrchr(path, '/');
#ifdef _WIN32
	const char *last2 = strrchr(path, '\\');
	if (!last1)
		return last2;

	if (!last2)
		return last1;

	return (last1 > last2) ? last1 : last2;
#else
	return last1;
#endif
}

bool path_directories_create(const char *path)
{
	if (!path || path[0] == '\0')
		return true;

	const char *last_sep = _find_last_sep(path);
	if (!last_sep)
		return true;

	size_t dir_len = (size_t)(last_sep - path);
	if (dir_len == 0)
		return true;

	char *dirbuf = malloc(dir_len + 1);
	if (!dirbuf)
		return false;

	memcpy(dirbuf, path, dir_len);
	dirbuf[dir_len] = '\0';

	char *p = dirbuf;
	if (p[0] == '/' || p[0] == '\\')
		p++;

	for (; *p; ++p) {
		if (*p == '/' || *p == '\\') {
			char saved = *p;
			*p = '\0';

			if (mkdir(dirbuf, 0755) != 0) {
				if (errno != EEXIST) {
					free(dirbuf);
					return false;
				}
			}

			*p = saved;
		}
	}

	if (mkdir(dirbuf, 0755) != 0) {
		if (errno != EEXIST) {
			free(dirbuf);
			return false;
		}
	}

	free(dirbuf);
	return true;
}

static int thread_num;

int arg_thread_num(void)
{
	if (!thread_num)
		thread_num = omp_get_max_threads();
	return thread_num;
}

static struct arg_callback parse_thread_num(const char *str, void *dest)
{
	errno = 0;
	char *endptr = NULL;
	unsigned long threads = strtoul(str, &endptr, 10);
	if (endptr == str || *endptr != '\0' || errno == ERANGE ||
	    threads > INT_MAX)
		return ARG_INVALID("Invalid thread count");

	if (threads) {
		*(int *)dest = (int)threads;
		omp_set_num_threads((int)threads);
	}

	return ARG_VALID();
}

static void print_threads(void)
{
	pinfol("CPU Threads: %d", arg_thread_num());
}

ARGUMENT(threads) = {
	.opt = 'T',
	.lopt = "threads",
	.help = "Number of threads (0 = auto)",
	.param = "N",
	.param_req = ARG_PARAM_REQUIRED,
	.dest = &thread_num,
	.parse_callback = parse_thread_num,
	.action_callback = print_threads,
	.action_weight = 1,
	.help_weight = 450,
};
