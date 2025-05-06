#pragma once
#ifndef PRINT_H
#define PRINT_H

#include <stddef.h>

#ifdef __cplusplus
#define P_RESTRICT __restrict
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define P_RESTRICT restrict
#else
#define P_RESTRICT
#endif

typedef enum
{
    FIRST,
    MIDDLE,
    LAST,
} location_t;

typedef struct
{
    char* ret;
    const size_t rsiz;
} input_t;

typedef const union
{
    const location_t loc;
    const int percent;
    char* const* choices;
    char** const* aliases;
    const input_t input;
} MSG_ARG;

#define MSG_LOC(location) ((MSG_ARG){ .loc = (location) })
#define MSG_PROPORTION(proportion) ((MSG_ARG){ .percent = ((int)(proportion * 100)) })
#define MSG_PERCENT(percentage) ((MSG_ARG){ .percent = ((int)(percentage)) })
#define MSG_CHOICE(choice_collection) ((MSG_ARG){ .choices = (choice_collection) })
#define MSG_ALIAS(alias_collection) ((MSG_ARG){ .aliases = (alias_collection) })
#define MSG_INPUT(result, rbuf_size) ((MSG_ARG){ .input = { .ret = result, .rsiz = rbuf_size } })
#define MSG_NONE MSG_LOC(FIRST)

typedef enum
{
    HEADER,
    SECTION,
    SUCCESS,
    INFO,
    VERBOSE,
    CONFIG,
    TIMING,
    DNA,
    PROGRESS,
    CHOICE,
    ALIAS,
    PROMPT,
    WARNING,
    ERROR,
    MSG_TYPE_COUNT
} message_t;

extern void print_verbose_flip();
extern void print_quiet_flip();

extern int print(message_t type, MSG_ARG margs, const char* P_RESTRICT format, ...);

/*
BASIC USAGE:

print(HEADER, MSG_NONE, "Header text");
╔══════════════════════════════════════════════════════════════════════════════╗
║                               Header text                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

print(SECTION, MSG_NONE, "Setup");
┌────────────────────────────────── Setup ───────────────────────────────────┐

print(SUCCESS, MSG_NONE, "Success text");
│ ✓ Success text                                                               │

const char* input_file = "input.csv";
print(INFO, MSG_NONE, "Reading input file: %s", input_file);
│ • Reading input file: input.csv                                              │

print(VERBOSE, MSG_NONE, "Batch size: %zu tasks per batch", optimal_batch_size);
│ · Batch size: 6163 tasks per batch                                           │

// Multiple related items with indentation hierarchy
print(CONFIG, MSG_LOC(FIRST), "Input: %s", input_file);
print(CONFIG, MSG_LOC(MIDDLE), "Output: %s", output_file);
print(CONFIG, MSG_LOC(LAST), "Compression: %d", compression_level);
│ ⚙ Input: in.csv                                                              │
│ ├ Output: out.h5                                                             │
│ └ Compression: 0                                                             │

// Timing breakdown with hierarchy
print(TIMING, MSG_LOC(FIRST), "Timing breakdown:");
print(TIMING, MSG_LOC(MIDDLE), "Init: %.3f sec (%.1f%%)", init_time, init_percent);
print(TIMING, MSG_LOC(MIDDLE), "Compute: %.3f sec (%.1f%%)", compute_time, compute_percent);
print(TIMING, MSG_LOC(MIDDLE), "I/O: %.3f sec (%.1f%%)", io_time, io_percent);
print(TIMING, MSG_LOC(LAST), "Total: %.3f sec (%.1f%%)", total_time, total_percent);
│ ⧗ Timing breakdown:                                                          │
│ ├ Init: 0.005 sec (7.5%)                                                     │
│ ├ Compute: 0.044 sec (73.0%)                                                 │
│ ├ I/O: 0.012 sec (19.4%)                                                     │
│ └ Total: 0.060 sec (100.0%)                                                  │

print(DNA, MSG_NONE, "Found %d sequences", seq_number);
│ ◇ Found 1042 sequences                                                       │

// Progress bar display
for (int i = 0; i < seq_number; i++) {
    int percentage = (i + 1) * 100 / seq_number;
    print(PROGRESS, MSG_PERCENT(percentage), "Storing sequences");
}

│ ▶ Storing sequences [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 100% │

// Interactive prompt with choices
char* choices[] = {"hello", "second", NULL};
int selected = print(CHOICE, MSG_CHOICE(choices), "Enter column number");
// This will display:
│ 1: hello                                                                     │
│ 2: second                                                                    │
│ • Enter column number (1-2): 4                                               │
│ ! Invalid input! Please enter a number between 1 and 2.                      │
// On valid input, returns the zero-based index of the selected choice

print(WARNING, MSG_NONE, "Warning text");
│ ! Warning text                                                               │

print(ERROR, MSG_NONE, "Error: File not found");
│ ✗ Error: File not found                                                      │

// Close section (useful for program exit, otherwise it will be closed automatically)
print(SECTION, MSG_NONE, NULL);
└──────────────────────────────────────────────────────────────────────────────┘
// For starting section with no text
print(SECTION, MSG_NONE, "");
┌──────────────────────────────────────────────────────────────────────────────┐
*/

#undef P_RESTRICT
#endif /* PRINT_H */