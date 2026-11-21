#ifndef UTIL_MACROS_H
#define UTIL_MACROS_H

#ifndef NDEBUG
#define unreachable_release() __asm__ volatile("int $0x03")
#else /* Release */
#define unreachable_release() __builtin_unreachable()
#endif /* NDEBUG */

#define byteof(ptr, nmemb) (sizeof(*(ptr)) * (nmemb))
#define alignments(n) (((n) * ((n) - 1)) >> 1)

#ifndef __cplusplus
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#endif /* UTIL_MACROS_H */
