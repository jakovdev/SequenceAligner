/* Compatibility header for systems that do not provide C11 threads.
 * Designed to be used for cross-compilation (mingw, msys2 ucrt64)
 */

#ifndef _THREADS_H_
#define _THREADS_H_

#include "sources/tinycthread.h"

#if defined(__GNUC__) && defined(__GNUC_MINOR__)
#define __GNUC_PREREQ(maj, min) ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
#define __GNUC_PREREQ(maj, min) 0
#endif

#if (!defined(__STDC_VERSION__) || __STDC_VERSION__ <= 201710L || !__GNUC_PREREQ(13, 0)) && !defined(__cplusplus)
#define thread_local _Thread_local
#endif

#endif /* _THREADS_H_ */
