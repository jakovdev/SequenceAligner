#pragma once
#ifndef PRINT_H
#define PRINT_H

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
// You can also freely customize icons, colors, return codes etc. in print.h and print.c.

print(HEADER, MSG_NONE, "Header text");
╔══════════════════════════════════════════════════════════════════════════════╗
║                                 Header text                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

print(SECTION, MSG_NONE, "Setup");
┌─────────────────────────────────── Setup ────────────────────────────────────┐

print(SUCCESS, MSG_NONE, "Success text");
│ ✓ Success text                                                               │

const char* input_file = "input.csv";
// Automatically formats the string like printf
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
    int percentage = 100 * (i + 1) / seq_number;
    print(PROGRESS, MSG_PERCENT(percentage), "Storing sequences");
}

// Has quick return for repeating percentages, draws over empty boxes
│ ▶ Storing sequences [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 100% │

// Interactive prompt with choices
char* choices[] = {"hello", "second"}; // Will auto NULL terminate, but you can also do it manually
int selected = print(CHOICE, MSG_CHOICE(choices, sizeof(choices)), "Enter column number");
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

// For getting user input
char result[16] = { 0 };
print(PROMPT, MSG_INPUT(result, sizeof(result)), "Enter a character: ");
│ • Enter a character: hello                                                   │
// result will now contain "hello"

// Quick y/N prompt (also has Y/n and y/n variants)
int answer = print_yN("Do you want to continue? (y/N): ");
│ • Do you want to continue? (y/N): y                                          │
// answer will be 1 (yes) or 0 (no)

// Close section (useful for program exit, otherwise it will be closed automatically)
print(SECTION, MSG_NONE, NULL);
└──────────────────────────────────────────────────────────────────────────────┘
// For starting section with no text
print(SECTION, MSG_NONE, "");
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

typedef enum
{
    FIRST,
    MIDDLE,
    LAST,
} p_location_t;

typedef struct
{
    char** chs;
    const int n;
} p_choice_t;

typedef struct
{
    char* ret;
    const int rsz;
} p_input_t;

typedef const union
{
    const p_location_t loc;
    const int percent;
    const p_choice_t choice_coll;
    const p_input_t input;
} MSG_ARG;

#define MSG_LOC(location) ((MSG_ARG){ .loc = (location) })
#define MSG_PROPORTION(proportion) ((MSG_ARG){ .percent = ((int)(proportion * 100)) })
#define MSG_PERCENT(percentage) ((MSG_ARG){ .percent = ((int)(percentage)) })
#define MSG_CHOICE(choices, cnum) ((MSG_ARG){ .choice_coll = { .chs = choices, .n = (int)cnum } })
#define MSG_INPUT(result, rsize) ((MSG_ARG){ .input = { .ret = result, .rsz = (int)rsize } })
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
    PROMPT,
    WARNING,
    ERROR,
    MSG_TYPE_COUNT
} message_t;

// Useful for debugging
typedef enum
{
    // All fields are customizable for easier debugging or if checks
    PRINT_SUCCESS = 0,
    PRINT_SKIPPED_BECAUSE_QUIET_OR_VERBOSE_NOT_ENABLED__SUCCESS = 0,
    PRINT_REPEAT_PROGRESS_PERCENT__SUCCESS = 0,
    PRINT_FIRST_CHOICE_INDEX__SUCCESS = 0, // Editable first choice index
    PRINT_INVALID_FORMAT_ARGS__ERROR = -1,
    PRINT_CHOICE_COLLECTION_SHOULD_CONTAIN_2_OR_MORE_CHOICES__ERROR = -2,
    PRINT_PROMPT_BUFFER_SIZE_SHOULD_BE_2_OR_MORE__ERROR = -2,
} print_return_t;

#define DEFINE_AS_1_TO_TURN_OFF_DEV_MESSAGES 0

extern void print_verbose_flip();
extern void print_quiet_flip();
extern void print_detail_flip();

extern print_return_t print(message_t type, MSG_ARG margs, const char* P_RESTRICT format, ...);

#define PRINT_USER_YES 1
#define PRINT_USER_NO 0

static inline int
print_yN(const char* P_RESTRICT prompt)
{
    char result[2] = { 0 };
    print(PROMPT, MSG_INPUT(result, sizeof(result)), prompt);
    if (result[0] == 'y' || result[0] == 'Y')
    {
        return PRINT_USER_YES;
    }

    else
    {
        return PRINT_USER_NO;
    }
}

static inline int
print_Yn(const char* P_RESTRICT prompt)
{
    char result[2] = { 0 };
    print(PROMPT, MSG_INPUT(result, sizeof(result)), prompt);
    if (result[0] == 'n' || result[0] == 'N')
    {
        return PRINT_USER_NO;
    }

    else
    {
        return PRINT_USER_YES;
    }
}

static inline int
print_yn(const char* P_RESTRICT prompt)
{
    char result[2] = { 0 };
repeat:
    print(PROMPT, MSG_INPUT(result, sizeof(result)), prompt);
    if (result[0] == 'y' || result[0] == 'Y')
    {
        return PRINT_USER_YES;
    }

    else if (result[0] == 'n' || result[0] == 'N')
    {
        return PRINT_USER_NO;
    }

    goto repeat;
}

#undef P_RESTRICT
#endif /* PRINT_H */