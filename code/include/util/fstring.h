#pragma once
#ifndef UTIL_FSTRING_H
#define UTIL_FSTRING_H

#include "system/arch.h"
#include "system/simd.h"

/*
Potential usages that need exploring:
sequences.h: sequence_init (memcpy)
csv.h: csv_line_parse (string copying)
TODO: Find more usages
*/

static inline char*
fast_strcpy(char* restrict dst, const char* restrict src, size_t len)
{
    size_t i = 0;
    uintptr_t src_align = (uintptr_t)src & (CACHE_LINE - 1);
    uintptr_t dst_align = (uintptr_t)dst & (CACHE_LINE - 1);

    if (len >= 64 && src_align == dst_align)
    {
        while (i < len && ((uintptr_t)(src + i) & (CACHE_LINE - 1)))
        {
            dst[i] = src[i];
            i++;
        }

#ifdef USE_SIMD
        for (; i <= len - BYTES; i += BYTES)
        {
            storeu((veci_t*)(dst + i), loadu((veci_t*)(src + i)));
        }

#endif
    }

    while (i < len)
    {
        dst[i] = src[i];
        i++;
    }

    return dst + len;
}

static inline char*
int_to_str(char* restrict str, int num)
{
    char* restrict ptr = str;
    uint32_t n = (num ^ (num >> 31)) - (num >> 31); // Convert to unsigned with abs
    uint32_t neg = (num >> 31) & 1;
    *ptr = '-';
    ptr += neg;

    // Buffer for digits (max 10 digits for 32-bit int + sign)
    char digits[12];
    char* restrict digit_ptr = digits;

    if (n == 0)
    {
        *ptr++ = '0';
        return ptr;
    }

    // Store digits in forward order
    do
    {
        uint32_t q = n / 10;
        uint32_t r = n - (q * 10); // mod
        *digit_ptr++ = (char)('0' + r);
        n = q;
    } while (n);

    // Copy digits in reverse order
    while (digit_ptr > digits)
    {
        *ptr++ = *--digit_ptr;
    }

    return ptr;
}

#endif // UTIL_FSTRING_H
