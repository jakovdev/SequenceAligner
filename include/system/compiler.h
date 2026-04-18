#ifndef SYSTEM_COMPILER_H
#define SYSTEM_COMPILER_H

#ifndef NDEBUG
#include <stdlib.h>
#define unreachable_release() abort()
#else /* Release */
#define unreachable_release() __builtin_unreachable()
#endif /* NDEBUG */

#define likely(x) (__builtin_expect(!!(x), 1))
#define unlikely(x) (__builtin_expect(!!(x), 0))

#ifndef __cplusplus
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#endif /* SYSTEM_COMPILER_H */
