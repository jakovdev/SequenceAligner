#pragma once
#ifndef UTIL_PRINT_H
#define UTIL_PRINT_H

/* REQUIRED: C99 or later */
/**BASIC USAGE:
 *
 * pheader("Header text");
 **╔══════════════════════════════════════════════════════════════════════════════╗
 **║                                 Header text                                  ║
 **╚══════════════════════════════════════════════════════════════════════════════╝
 *
 * psection("Setup");
 **┌─────────────────────────────────── Setup ────────────────────────────────────┐
 *
 **Automatically formats the string like printf
 * const char *input_file = "input.csv";
 * pinfo("Reading input file: %s", input_file);
 **│ • Reading input file: input.csv                                              │
 *
 **Only prints if verbose mode is enabled
 * pverb("Batch size: %zu tasks per batch", batch_size);
 **│ · Batch size: 6163 tasks per batch                                           │
 *
 **Specify location for simple hierarchy
 * pinfo("Input: %s", input_file);
 * pinfom("Output: %s", output_file);
 * pinfol("Compression: %d", compression_level);
 **│ • Input: in.csv                                                              │
 **│ ├ Output: out.h5                                                             │
 **│ └ Compression: 0                                                             │
 *
 **Progress bar display, has quick return for repeats, draws over empty boxes
 * int seq_number = 1000;
 * for (int i = 0; i < seq_number; i++)
 *     pproport(i / seq_number, "Storing sequences");
 **│ ▶ Storing sequences [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■·······]  86% │
 *
 **Interactive prompt with choices
 * char *choices[] = {"hello", "second", NULL};
 * int selected = pchoice_s(choices, "Enter column number");
 **│ 1: hello                                                                     │
 **│ 2: second                                                                    │
 **│ • Enter column number (1-2): 4                                               │
 **│ ! Invalid input! Please enter a number between 1 and 2.                      │
 **On valid input, returns the zero-based index of the selected choice
 *
 * pwarn("Warning text");
 **│ ! Warning text                                                               │
 *
 * perror_context("FILES");
 * perror("File not found");
 **│ ✗ FILES | File not found                                                     │
 *
 * perror_context(NULL);
 * perror("File not found");
 **│ ✗ File not found                                                             │
 *
 **For getting user input
 * char result[16] = { 0 };
 * print(M_IS(result) "Enter a character: ");
 **│ • Enter a character: hello                                                   │
 **result will now contain "hello"
 *
 **Quick y/N prompt (also has Y/n and y/n variants)
 * bool answer = print_yN("Do you want to continue?");
 **│ • Do you want to continue? [y/N]: y                                          │
 **answer will be true (yes) or false (no)
 *
 * pdev("Error: %d", value);
 **│ ✗ _TO_DEV_ | Error: 42                                                       │
 **Only prints in debug builds (NDEBUG not defined) with error context _TO_DEV_
 *
 **Close section (automatic on new section/header or non-abort exit)
 * psection_end();
 **└──────────────────────────────────────────────────────────────────────────────┘
 *
 **For starting section with no text
 * psection();
 **┌──────────────────────────────────────────────────────────────────────────────┐
 *
 **Also available:
 * print_streams(stdin, stdout, stderr);
 */

#ifdef __cplusplus
#if defined(__GNUC__) || defined(__clang__)
#define P_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define P_RESTRICT __restrict
#else
#define P_RESTRICT
#endif

#else
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define P_RESTRICT restrict
#else
/* While restrict is optional, snprintf and vsnprintf are required */
#error "Compiler does not support C99 or later. Please use a compatible compiler."
#endif
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>

enum p_location {
	LOC_FIRST,
	LOC_MIDDLE,
	LOC_LAST,
#define _M_LOC(l) ((M_ARG){ .loc = (l) })
};

struct p_choice {
	char **choices;
	const size_t n;
#define _M_CHOICE(c, s) \
	((M_ARG){ .choice = { .choices = (c), .n = (s) } }), P_CHOICE
};

struct p_input {
	char *out;
	const size_t out_size;
#define _M_INPUT(o, s) \
	((M_ARG){ .input = { .out = (o), .out_size = (s) } }), P_INPUT
};

typedef const union {
	const enum p_location loc;
	const int percent;
#define _M_PERCENT(p) ((M_ARG){ .percent = (p) }), P_PROGRESS
	const struct p_choice choice;
	const struct p_input input;
} M_ARG;

#define M_LOC(location) _M_LOC(location)

#define M_PERCENT(percentage) _M_PERCENT(percentage)
#define M_PROPORT(proportion) _M_PERCENT((int)(100 * proportion))

#define M_CHOICE(choices, n) _M_CHOICE(choices, n)
#define M_CS(choices) _M_CHOICE(choices, sizeof(choices))

#define M_INPUT(out, s) _M_INPUT(out, s)
#define M_IS(out) _M_INPUT(out, sizeof(out))

#define M_NONE ((M_ARG){ 0 })

/* clang-format off */
#define P_NONE	   "0" /* No special message type */
#define P_INFO     "1" /* Regular info message */
#define P_VERBOSE  "2" /* Info message controlled by verbose flag */
#define P_WARNING  "3" /* Warning message, something is ignored, assumption */
#define P_ERROR    "4" /* Error message, reason why program has to exit */
#define P_CHOICE   "5" /* User choice prompt, numbered range */
#define P_INPUT    "6" /* User input prompt, free text */
#define P_PROGRESS "7" /* Progress bar display */
#define P_HEADER   "8" /* Header box, for large titles */
#define P_SECTION  "9" /* Section box, for separating by context */
/* clang-format on */

/* Useful for debugging */
enum p_return {
	/* All fields are customizable for easier debugging or if checks */
	PRINT_SUCCESS = 0,
	PRINT_SKIPPED_BECAUSE_QUIET_OR_VERBOSE_NOT_ENABLED__SUCCESS = 0,
	PRINT_REPEAT_PROGRESS_PERCENT__SUCCESS = 0,
	PRINT_FIRST_CHOICE_INDEX__SUCCESS = 0, /* Editable first choice index */
	PRINT_INVALID_FORMAT_ARGS__ERROR = -1,
	PRINT_CHOICE_COLLECTION_SHOULD_CONTAIN_2_OR_MORE_CHOICES__ERROR = -2,
	PRINT_INPUT_BUFFER_SIZE_SHOULD_BE_2_OR_MORE__ERROR = -2,
	PRINT_TO_DEV_NDEBUG__ERROR = -0xDEAD
};

void print_streams(FILE *in, FILE *out, FILE *err);

enum p_return print(M_ARG, const char *P_RESTRICT fmt, ...);

/* "prompt [y/N]: " */
bool print_yN(const char *P_RESTRICT prompt);
/* "prompt [Y/n]: " */
bool print_Yn(const char *P_RESTRICT prompt);
/* "prompt [y/n]: " */
bool print_yn(const char *P_RESTRICT prompt);

#define pinfo(...) print(M_NONE, P_INFO __VA_ARGS__)
#define pinfom(...) print(M_LOC(LOC_MIDDLE), P_INFO __VA_ARGS__)
#define pinfol(...) print(M_LOC(LOC_LAST), P_INFO __VA_ARGS__)

#define pwarning(...) print(M_NONE, P_WARNING __VA_ARGS__)
#define pwarningm(...) print(M_LOC(LOC_MIDDLE), P_WARNING __VA_ARGS__)
#define pwarningl(...) print(M_LOC(LOC_LAST), P_WARNING __VA_ARGS__)

#define pwarn(...) pwarning(__VA_ARGS__)
#define pwarnm(...) pwarningm(__VA_ARGS__)
#define pwarnl(...) pwarningl(__VA_ARGS__)

#define pverbose(...) print(M_NONE, P_VERBOSE __VA_ARGS__)
#define pverbosem(...) print(M_LOC(LOC_MIDDLE), P_VERBOSE __VA_ARGS__)
#define pverbosel(...) print(M_LOC(LOC_LAST), P_VERBOSE __VA_ARGS__)

#define pverb(...) pverbose(__VA_ARGS__)
#define pverbm(...) pverbosem(__VA_ARGS__)
#define pverbl(...) pverbosel(__VA_ARGS__)

void perror_context(const char *prefix);
#define perror(...) print(M_NONE, P_ERROR __VA_ARGS__)
#define perrorm(...) print(M_LOC(LOC_MIDDLE), P_ERROR __VA_ARGS__)
#define perrorl(...) print(M_LOC(LOC_LAST), P_ERROR __VA_ARGS__)

#define pchoice(choices, n, ...) print(M_CHOICE(choices, n) __VA_ARGS__)
#define pchoice_s(choices, ...) print(M_CS(choices) __VA_ARGS__)
#define pinput(out, size, ...) print(M_INPUT(out, size) __VA_ARGS__)
#define pinput_s(out, ...) print(M_IS(out) __VA_ARGS__)

#define ppercent(pct, ...) print(M_PERCENT(pct) __VA_ARGS__)
#define pproport(prp, ...) print(M_PROPORT(prp) __VA_ARGS__)

#define pheader(...) print(M_NONE, P_HEADER __VA_ARGS__)

#define psection(...) print(M_NONE, P_SECTION __VA_ARGS__)
#define psection_end() print(M_NONE, NULL)

/* Doesn't print in RELEASE (NDEBUG defined), _TO_DEV_ error context */
enum p_return pdev(const char *fmt, ...);

#undef P_RESTRICT
#endif /* UTIL_PRINT_H */
