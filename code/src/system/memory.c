#include "system/memory.h"

#include <string.h>
#include <stdlib.h>
#include <stddef.h>

#ifndef _WIN32
#include <stdio.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#else
#include "system/os.h"
#endif

#if __STDC_VERSION__ >= 201112L || defined(_WIN32)
#define ALIGNED_ALLOC_AVAILABLE
#endif

#define HUGE_PAGE (2 * MiB)

#include "system/types.h"

void *alloc_huge_page(size_t size)
{
	void *ptr = NULL;
#ifdef __linux__
	if (size >= HUGE_PAGE) {
		size_t aligned = (size + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1);
#ifdef ALIGNED_ALLOC_AVAILABLE
		ptr = aligned_alloc(HUGE_PAGE, aligned);
#else
		if (posix_memalign(&ptr, HUGE_PAGE, aligned) != 0)
			ptr = NULL;
#endif
		if (ptr) {
			madvise(ptr, size, MADV_HUGEPAGE);
			return ptr;
		}
	}

#endif

	size_t aligned = (size + CACHE_LINE - 1) & ~(CACHE_LINE - 1);
#ifdef ALIGNED_ALLOC_AVAILABLE
	ptr = aligned_alloc(CACHE_LINE, aligned);
#else
	if (posix_memalign(&ptr, CACHE_LINE, aligned) != 0)
		ptr = NULL;
#endif

	return ptr;
}

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
			ull val = strtoull(line + 13, &endptr, 10);
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
		if (sysinfo(&info) == 0) {
			available_mem = info.freeram * info.mem_unit;
		}
	}

#endif

	return available_mem;
}
