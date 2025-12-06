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
 * perr_context("FILES");
 * perr("File not found");
 **│ ✗ FILES | File not found                                                     │
 *
 * perr_context(NULL);
 * perr("File not found");
 **│ ✗ File not found                                                             │
 *
 **For getting user input
 * char result[16] = { 0 };
 * pinput_s(result, "Enter a character: ");
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

typedef const union {
	char **choices;
#define P_INPUT_C(c, n) (P_INPUT){ .choices = c }, n, P_CHOICE
#define P_INPUT_CS(choices) P_INPUT_C(choices, sizeof(choices))
	char *output;
#define P_INPUT_P(out, size) (P_INPUT){ .output = (out) }, size, P_PROMPT
#define P_INPUT_PS(out) P_INPUT_P(out, sizeof(out))
} P_INPUT;

/* clang-format off */
#define P_INFO    "\x01"
#define P_VERBOSE "\x02"
#define P_WARNING "\x03"
#define P_ERROR   "\x04"
#define P_HEADER  "\x05"
#define P_SECTION "\x06"
#define P_CHOICE  "\x07"
#define P_PROMPT  "\x08"
#define P_MIDDLE  "\x11"
#define P_LAST    "\x12"
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
	PRINT_PROMPT_BUFFER_SIZE_SHOULD_BE_2_OR_MORE__ERROR = -2,
	PRINT_INVALID_INPUT_TYPE__ERROR = -3,
	PRINT_TO_DEV_NDEBUG__ERROR = -0xDEAD
};

void print_streams(FILE *in, FILE *out, FILE *err);

enum p_return print(const char *P_RESTRICT fmt, ...);
enum p_return progress_bar(int percent, const char *P_RESTRICT fmt, ...);
enum p_return input(P_INPUT, size_t, const char *P_RESTRICT fmt, ...);

/* "prompt [y/N]: " */
bool print_yN(const char *P_RESTRICT prompt);
/* "prompt [Y/n]: " */
bool print_Yn(const char *P_RESTRICT prompt);
/* "prompt [y/n]: " */
bool print_yn(const char *P_RESTRICT prompt);

#define pinfo(...) print(P_INFO __VA_ARGS__)
#define pinfom(...) print(P_INFO P_MIDDLE __VA_ARGS__)
#define pinfol(...) print(P_INFO P_LAST __VA_ARGS__)

#define pwarning(...) print(P_WARNING __VA_ARGS__)
#define pwarningm(...) print(P_WARNING P_MIDDLE __VA_ARGS__)
#define pwarningl(...) print(P_WARNING P_LAST __VA_ARGS__)

#define pwarn(...) pwarning(__VA_ARGS__)
#define pwarnm(...) pwarningm(__VA_ARGS__)
#define pwarnl(...) pwarningl(__VA_ARGS__)

#define pverbose(...) print(P_VERBOSE __VA_ARGS__)
#define pverbosem(...) print(P_VERBOSE P_MIDDLE __VA_ARGS__)
#define pverbosel(...) print(P_VERBOSE P_LAST __VA_ARGS__)

#define pverb(...) pverbose(__VA_ARGS__)
#define pverbm(...) pverbosem(__VA_ARGS__)
#define pverbl(...) pverbosel(__VA_ARGS__)

void perr_context(const char *prefix);
#define perr(...) print(P_ERROR __VA_ARGS__)
#define perrm(...) print(P_ERROR P_MIDDLE __VA_ARGS__)
#define perrl(...) print(P_ERROR P_LAST __VA_ARGS__)

#define pchoice(choices, n, ...) input(P_INPUT_C(choices, n) __VA_ARGS__)
#define pchoice_s(choices, ...) input(P_INPUT_CS(choices) __VA_ARGS__)
#define pinput(out, size, ...) input(P_INPUT_P(out, size) __VA_ARGS__)
#define pinput_s(out, ...) input(P_INPUT_PS(out) __VA_ARGS__)

#define ppercent(pct, ...) progress_bar(pct, __VA_ARGS__)
#define pproport(prp, ...) progress_bar((100 * prp), __VA_ARGS__)
#define pproportc(prp, ...) progress_bar((int)(100 * prp), __VA_ARGS__)

#define pheader(...) print(P_HEADER __VA_ARGS__)

#define psection(...) print(P_SECTION __VA_ARGS__)
#define psection_end() print(NULL)

/* Doesn't print in RELEASE (NDEBUG defined), _TO_DEV_ error context */
enum p_return pdev(const char *fmt, ...);

#undef P_RESTRICT
#endif /* UTIL_PRINT_H */
