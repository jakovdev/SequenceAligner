#ifndef UTIL_MACROS_H
#define UTIL_MACROS_H

#ifndef NDEBUG
#include <stdlib.h>
#define unreachable_release() abort()
#else /* Release */
#define unreachable_release() __builtin_unreachable()
#endif /* NDEBUG */

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define sizeof_field(t, f) (sizeof(((t *)0)->f))
#define bytesof(ptr, nmemb) (sizeof(*(ptr)) * nmemb)

#ifndef __cplusplus
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#endif /* UTIL_MACROS_H */
