#ifndef ARCH_H
#define ARCH_H

// TODO: Expand

// Helper constants, do not change //
#define KiB (1ULL << 10)
#define MiB (KiB << 10)
#define GiB (MiB << 10)

#define USE_HUGE_PAGES 1
#define HUGE_PAGE_THRESHOLD (2 * MiB)

#define CACHE_LINE 64
#define L1_CACHE_SIZE (32 * KiB)  // Typical L1 cache size
#define L2_CACHE_SIZE (256 * KiB) // Typical L2 cache size
#define MAX_THREADS (32)

#define MAX_STACK_SEQUENCE_LENGTH (4 * KiB)

// CPU architecture detection
#if defined(__x86_64__) || defined(_M_X64)
#define ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ARCH_ARM64 1
#endif

#if defined(ARCH_X86_64)
#define PREFETCH_DISTANCE 16
#elif defined(ARCH_ARM64)
#define PREFETCH_DISTANCE 8
#else
#define PREFETCH_DISTANCE 4
#endif

#endif