#pragma once
#ifndef UTIL_PRINT_H
#define UTIL_PRINT_H

/*
REQUIRED: C99 or later
BASIC USAGE:

// Setup box width:
#define TERMINAL_WIDTH 80
// Will automatically adjust to this terminal width

// Also available:
print_quiet_flip(); // Only prints important messages like errors and user prompts
print_detail_flip(); // Prints the message without the details (box, icon, etc.)
print_verbose_flip(); // Prints VERBOSE messages
print_streams(stdin, stdout, stderr); // Set input/output/error streams
print_error_context("ARGS"); // Will be prepended to error messages
// You can also freely customize icons, colors, return codes etc. in print.h and print.c.

print(M_NONE, HEADER "Header text");
╔══════════════════════════════════════════════════════════════════════════════╗
║                                 Header text                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

print(M_NONE, SECTION "Setup");
┌─────────────────────────────────── Setup ────────────────────────────────────┐
const char* input_file = "input.csv";
// Automatically formats the string like printf
print(M_NONE, INFO "Reading input file: %s", input_file);
│ • Reading input file: input.csv                                              │
print(M_NONE, VERBOSE "Batch size: %zu tasks per batch", optimal_batch_size);
│ · Batch size: 6163 tasks per batch                                           │
// Multiple related items with indentation hierarchy
print(M_LOC(FIRST), INFO "Input: %s", input_file);
print(M_LOC(MIDDLE), INFO "Output: %s", output_file);
print(M_LOC(LAST), INFO "Compression: %d", compression_level);
│ • Input: in.csv                                                              │
│ ├ Output: out.h5                                                             │
│ └ Compression: 0                                                             │
print(M_LOC(FIRST), INFO "Timing breakdown:");
print(M_LOC(MIDDLE), INFO "Init: %.3f sec (%.1f%%)", init_time, init_percent);
print(M_LOC(MIDDLE), INFO "Compute: %.3f sec (%.1f%%)", compute_time, compute_percent);
print(M_LOC(MIDDLE), INFO "I/O: %.3f sec (%.1f%%)", io_time, io_percent);
print(M_LOC(LAST), INFO "Total: %.3f sec (%.1f%%)", total_time, total_percent);
│ • Timing breakdown:                                                          │
│ ├ Init: 0.005 sec (7.5%)                                                     │
│ ├ Compute: 0.044 sec (73.0%)                                                 │
│ ├ I/O: 0.012 sec (19.4%)                                                     │
│ └ Total: 0.060 sec (100.0%)                                                  │
// Progress bar display
int seq_number = 1000;
for (int i = 0; i < seq_number; i++)
    print(M_PROPORT(i / seq_number) "Storing sequences");
// Has quick return for repeating percentages, draws over empty boxes
│ ▶ Storing sequences [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 100% │
// Interactive prompt with choices
char* choices[] = {"hello", "second", NULL}; // Will auto NULL terminate, but you should do it manually
int selected = print(M_CS(choices), "Enter column number");
// This will display:
│ 1: hello                                                                     │
│ 2: second                                                                    │
│ • Enter column number (1-2): 4                                               │
│ ! Invalid input! Please enter a number between 1 and 2.                      │
// On valid input, returns the zero-based index of the selected choice
print(M_NONE, WARNING "Warning text");
│ ! Warning text                                                               │
print_error_context("FILES");
print(M_NONE, ERR "File not found");
│ ✗ FILES | File not found                                                     │
print_error_context(NULL);
print(M_NONE, ERR "File not found");
│ ✗ File not found                                                             │
// For getting user input
char result[16] = { 0 };
print(M_UINS(result) "Enter a character: ");
│ • Enter a character: hello                                                   │
// result will now contain "hello"
// Quick y/N prompt (also has Y/n and y/n variants)
bool answer = print_yN("Do you want to continue?");
│ • Do you want to continue? [y/N]: y                                          │
// answer will be true (yes) or false (no)
// Close section (will do it automatically if possible (exit, new section/header))
// print(M_NONE, NULL);
└──────────────────────────────────────────────────────────────────────────────┘
// For starting section with no text
print(M_NONE, SECTION);
┌──────────────────────────────────────────────────────────────────────────────┐
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
	FIRST,
	MIDDLE,
	LAST,
#define _M_LOC(l) ((M_ARG){ .loc = (l) })
};

struct p_choice {
	char **choices;
	const size_t n;
#define _M_CHOICE(c, s) \
	((M_ARG){ .choice = { .choices = (c), .n = (s) } }), CHOICE
};

struct p_uinput {
	char *out;
	const size_t out_size;
#define _M_UINPUT(o, s) \
	((M_ARG){ .uinput = { .out = (o), .out_size = (s) } }), PROMPT
};

typedef const union {
	const enum p_location loc;
	const int percent;
#define _M_PERCENT(p) ((M_ARG){ .percent = (p) }), PROGRESS
	const struct p_choice choice;
	const struct p_uinput uinput;
} M_ARG;

#define M_LOC(location) _M_LOC(location)

#define M_PERCENT(percentage) _M_PERCENT(percentage)
#define M_PROPORT(proportion) _M_PERCENT((int)(100 * proportion))

#define M_CHOICE(choices, n) _M_CHOICE(choices, n)
#define M_CS(choices) _M_CHOICE(choices, sizeof(choices))

#define M_UINPUT(out, s) _M_UINPUT(out, s)
#define M_UINS(out) _M_UINPUT(out, sizeof(out))

#define M_NONE M_LOC(FIRST)

// clang-format off
#define NONE	 "0" /* No special message type */
#define INFO     "1" /* Regular info message */
#define VERBOSE  "2" /* Info message controlled by verbose flag */
#define WARNING  "3" /* Warning message, something is ignored, assumption */
#define ERR      "4" /* Error message, reason why program has to exit */
#define CHOICE   "5" /* User choice prompt, numbered range */
#define PROMPT   "6" /* User input prompt, free text */
#define PROGRESS "7" /* Progress bar display */
#define HEADER   "8" /* Header box, for large titles */
#define SECTION  "9" /* Section box, for separating by context */
// clang-format on

// Useful for debugging
enum p_return {
	/* All fields are customizable for easier debugging or if checks */
	PRINT_SUCCESS = 0,
	PRINT_SKIPPED_BECAUSE_QUIET_OR_VERBOSE_NOT_ENABLED__SUCCESS = 0,
	PRINT_REPEAT_PROGRESS_PERCENT__SUCCESS = 0,
	PRINT_FIRST_CHOICE_INDEX__SUCCESS = 0, // Editable first choice index
	PRINT_INVALID_FORMAT_ARGS__ERROR = -1,
	PRINT_CHOICE_COLLECTION_SHOULD_CONTAIN_2_OR_MORE_CHOICES__ERROR = -2,
	PRINT_PROMPT_BUFFER_SIZE_SHOULD_BE_2_OR_MORE__ERROR = -2,
	PRINT_TO_DEV_NDEBUG__ERROR = -0xDEAD
};

void print_verbose_flip(void);
void print_quiet_flip(void);
void print_detail_flip(void);

void print_error_context(const char *prefix);

void print_streams(FILE *in, FILE *out, FILE *err);

enum p_return print(M_ARG, const char *P_RESTRICT format, ...);
bool print_yN(const char *P_RESTRICT prompt);
bool print_Yn(const char *P_RESTRICT prompt);
bool print_yn(const char *P_RESTRICT prompt);

/* Doesn't print in RELEASE (NDEBUG defined), _TO_DEV_ error context */
enum p_return print_dev(const char *fmt, ...);

#undef P_RESTRICT
#endif /* UTIL_PRINT_H */
