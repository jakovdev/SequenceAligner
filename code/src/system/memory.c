#include "system/memory.h"

#ifdef _WIN32
#include "system/os.h"
#else
#include <stdio.h>
#include <string.h>
#include <sys/sysinfo.h>
#endif

#include "system/types.h"

void *alloc_aligned(size_t alignment, size_t bytes)
{
	if (alignment < sizeof(void *) || (alignment & (alignment - 1)) != 0)
		return NULL;

	void *ptr = NULL;
#if __STDC_VERSION__ >= 201112L && !defined(__APPLE__)
	if (bytes % alignment != 0)
		bytes = (bytes + alignment - 1) & ~(alignment - 1);

	ptr = aligned_alloc(alignment, bytes);
#else
	if (posix_memalign(&ptr, alignment, bytes) != 0)
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
