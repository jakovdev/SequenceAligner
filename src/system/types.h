#ifndef SYSTEM_TYPES_H
#define SYSTEM_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* scores, sequence lengths, offsets and counts */
typedef int32_t s32;
constexpr s32 S32_MAX = INT32_MAX;
constexpr s32 S32_MIN = INT32_MIN;
/* alignment sizes, products */
typedef int64_t s64;
constexpr s64 S64_MAX = INT64_MAX;
constexpr s64 S64_MIN = INT64_MIN;
/* sequence letters */
typedef uint8_t u8;

#ifdef __cplusplus
#define restrict __restrict__
#endif

#ifndef _WIN32
#define SECTION(type, name) aligned(alignof(type)), section(name), used, retain
#define ASM_SECTION_PUSH(section) asm(".pushsection ." section ",\"a\"\n\t")
#define ASM_SECTION_POP(section) asm(".popsection\n\t")
#else
#define SECTION(type, name) aligned(alignof(type)), section(name), used
#define ASM_SECTION_PUSH(section) asm(".section ." section ",\"r\"\n\t")
#define ASM_SECTION_POP(section) asm(".section ." section "\n\t")
#endif

#define ASM_SECTION(into, section) \
	ASM_SECTION_PUSH(section); \
	asm(into);                 \
	ASM_SECTION_POP(section)
#define ROSTRING_CREATE(name, ...)                                     \
	extern const char name[];                                      \
	ASM_SECTION(".globl " #name "\n\t" #name ":\n\t", #name "$A"); \
	__VA_OPT__(ROSTRING_EXTEND(name, __VA_ARGS__);)                \
	ASM_SECTION(".byte 0\n\t", #name "$C")
#define ROSTRING_EXTEND(name, str) \
	ASM_SECTION(".ascii \"" str "\"\n\t", #name "$B")

#endif /* SYSTEM_TYPES_H */
