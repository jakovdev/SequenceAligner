#ifndef ARCH_H
#define ARCH_H

// TODO: Expand

// Compile-time memory management optimizations
#define USE_HUGE_PAGES 1  // Use huge pages for large allocations
#define CACHE_LINE 64
#define L1_CACHE_SIZE (32 * 1024)  // Typical L1 cache size
#define L2_CACHE_SIZE (256 * 1024) // Typical L2 cache size

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