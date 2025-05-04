#ifndef PRINT_H
#define PRINT_H

#include "arch.h"
#include "terminal.h"
#include <stdarg.h>
#include <stdbool.h>

#ifndef TERMINAL_WIDTH
#define TERMINAL_WIDTH 80
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

typedef enum
{
    COLOR_RESET,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_BLUE,
    COLOR_MAGENTA,
    COLOR_CYAN,
    COLOR_GRAY,
    COLOR_BRIGHT_CYAN,
    COLOR_TYPE_COUNT
} color_t;

typedef enum
{
    ICON_NONE,
    ICON_INFO,
    ICON_SUCCESS,
    ICON_WARNING,
    ICON_ERROR,
    ICON_CLOCK,
    ICON_DNA,
    ICON_GEAR,
    ICON_ARROW,
    ICON_DOT,
    ICON_TYPE_COUNT
} icon_t;

typedef enum
{
    OPTIONAL,
    REQUIRED
} requirement_t;

enum
{
    BOX_TOP_LEFT,
    BOX_LEFT_TEE,
    BOX_BOTTOM_LEFT,
    BOX_TOP_RIGHT,
    BOX_HORIZONTAL,
    BOX_VERTICAL,
    BOX_RIGHT_TEE,
    BOX_BOTTOM_RIGHT,
    BOX_CHAR_COUNT
};

enum
{
    BOX_NORMAL,
    BOX_FANCY,
    BOX_TYPE_COUNT
};

static struct
{
    struct
    {
        const char* codes[COLOR_TYPE_COUNT];
        const char* icons[ICON_TYPE_COUNT];
        const char* boxes[BOX_TYPE_COUNT][BOX_CHAR_COUNT];
        const char* progress_filled_char;
        const char* progress_empty_char;
        const char* ansi_escape_start;
        const char* ansi_carriage_return;
    } chars;

    const struct
    {
        color_t color;
        icon_t icon;
        requirement_t requirement;
    } map[MSG_TYPE_COUNT];

    size_t total_width;

    struct
    {
        unsigned verbose : 1;
        unsigned quiet : 1;
        unsigned section_open : 1;
        unsigned content_printed : 1;
    } flags;

} style = {

    .chars = {
        .codes = {
            [COLOR_RESET]       = "\x1b[0m",
            [COLOR_RED]         = "\x1b[31m",
            [COLOR_GREEN]       = "\x1b[32m",
            [COLOR_YELLOW]      = "\x1b[33m",
            [COLOR_BLUE]        = "\x1b[34m",
            [COLOR_MAGENTA]     = "\x1b[35m",
            [COLOR_CYAN]        = "\x1b[36m",
            [COLOR_GRAY]        = "\x1b[90m",
            [COLOR_BRIGHT_CYAN] = "\x1b[96m",
        },
        .icons = {
            [ICON_NONE]    = "",
            [ICON_INFO]    = "•",
            [ICON_SUCCESS] = "✓",
            [ICON_WARNING] = "!",
            [ICON_ERROR]   = "✗",
            [ICON_CLOCK]   = "⧗",
            [ICON_DNA]     = "◇",
            [ICON_GEAR]    = "⚙",
            [ICON_ARROW]   = "▶",
            [ICON_DOT]     = "·",
        },
        .boxes = {
            {
                "┌", "├", "└", "┐", "─", "│", "┤", "┘"
            },
            {
                "╔", "╠", "╚", "╗", "═", "║", "╣", "╝"
            },
        },
        .progress_filled_char = "■",
        .progress_empty_char = "·",
        .ansi_escape_start = "\x1b",
        .ansi_carriage_return = "\r",
    },
    
    .map = {
        [HEADER]   = { COLOR_BRIGHT_CYAN, ICON_NONE,    OPTIONAL },
        [SECTION]  = { COLOR_BLUE,        ICON_NONE,    OPTIONAL },
        [SUCCESS]  = { COLOR_GREEN,       ICON_SUCCESS, OPTIONAL },
        [INFO]     = { COLOR_BLUE,        ICON_INFO,    OPTIONAL },
        [VERBOSE]  = { COLOR_GRAY,        ICON_DOT,     REQUIRED },
        [CONFIG]   = { COLOR_YELLOW,      ICON_GEAR,    OPTIONAL },
        [TIMING]   = { COLOR_CYAN,        ICON_CLOCK,   REQUIRED },
        [DNA]      = { COLOR_MAGENTA,     ICON_DNA,     OPTIONAL },
        [PROGRESS] = { COLOR_BRIGHT_CYAN, ICON_ARROW,   OPTIONAL },
        [CHOICE]   = { COLOR_BLUE,        ICON_INFO,    REQUIRED },
        [ALIAS]    = { COLOR_BLUE,        ICON_INFO,    REQUIRED },
        [PROMPT]   = { COLOR_BLUE,        ICON_INFO,    REQUIRED },
        [WARNING]  = { COLOR_YELLOW,      ICON_WARNING, REQUIRED },
        [ERROR]    = { COLOR_RED,         ICON_ERROR,   REQUIRED },
    },
    
    .total_width = TERMINAL_WIDTH,
};

static inline void
print_verbose_flip()
{
    style.flags.verbose = !style.flags.verbose;
}

static inline void
print_quiet_flip()
{
    style.flags.quiet = !style.flags.quiet;
}

static inline void
print_context_init()
{
    terminal_init();
    if (!terminal_environment())
    {
        style.chars.ansi_carriage_return = "\n";
    }
}

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

static int
print(message_t type, MSG_ARG margs, const char* restrict format, ...)
{
    const bool is_required = style.map[type].requirement == REQUIRED;
    if ((style.flags.quiet && !is_required) || (type == VERBOSE && !style.flags.verbose))
    {
        return 0;
    }

    if (!style.flags.section_open && type != HEADER && type != SECTION)
    {
        print(SECTION, MSG_NONE, "");
    }

    static int last_percentage = -1;
    if (type == PROGRESS)
    {
        if (margs.percent == last_percentage)
        {
            return 0;
        }

        last_percentage = margs.percent;

        if (last_percentage == 100)
        {
            last_percentage = -1;
        }
    }

    else if (type == ERROR && last_percentage != -1)
    {
        if (style.flags.content_printed)
        {
            printf("\n");
            last_percentage = -1;
        }
    }

    const bool simple_format = style.flags.quiet && is_required;
    const icon_t icon_type = style.map[type].icon;

    const char* c_icon = style.chars.icons[icon_type];
    const char* c_color = style.chars.codes[style.map[type].color];

    const char* section_color = style.chars.codes[style.map[SECTION].color];
    const char* reset_code = style.chars.codes[COLOR_RESET];
    const char* box_vertical = style.chars.boxes[BOX_NORMAL][BOX_VERTICAL];

    const size_t box_char_width = simple_format ? 0 : 1;
    const size_t icon_width = (simple_format || icon_type == ICON_NONE) ? 0 : 2;
    const size_t available = style.total_width - (2 * box_char_width) - icon_width - 1;

    char buffer[BUFSIZ] = { 0 };
    int buflen = 0;

    va_list args;
    va_start(args, format);

    if (format != NULL)
    {
        buflen = vsnprintf(buffer, sizeof(buffer), format, args);
        if (buflen < 0)
        {
            va_end(args);
            return -1;
        }
    }

    if (type == HEADER)
    {
        if (format == NULL)
        {
            goto cleanup;
        }

        if (style.flags.section_open)
        {
            print(SECTION, MSG_NONE, NULL);
        }

        // Top border
        printf("%s%s", c_color, style.chars.boxes[BOX_FANCY][BOX_TOP_LEFT]);
        for (size_t i = 0; i < style.total_width - 2; i++)
        {
            printf("%s", style.chars.boxes[BOX_FANCY][BOX_HORIZONTAL]);
        }

        printf("%s%s\n", style.chars.boxes[BOX_FANCY][BOX_TOP_RIGHT], reset_code);

        // Content with centering
        const size_t left_padding = (style.total_width - 2 - buflen) / 2;
        const size_t right_padding = style.total_width - 2 - buflen - left_padding;

        printf("%s%s", c_color, style.chars.boxes[BOX_FANCY][BOX_VERTICAL]);
        printf("%*s%s%*s", (int)left_padding, "", buffer, (int)right_padding, "");
        printf("%s%s\n", style.chars.boxes[BOX_FANCY][BOX_VERTICAL], reset_code);

        // Bottom border
        printf("%s%s", c_color, style.chars.boxes[BOX_FANCY][BOX_BOTTOM_LEFT]);
        for (size_t i = 0; i < style.total_width - 2; i++)
        {
            printf("%s", style.chars.boxes[BOX_FANCY][BOX_HORIZONTAL]);
        }

        printf("%s%s\n", style.chars.boxes[BOX_FANCY][BOX_BOTTOM_RIGHT], reset_code);

        style.flags.section_open = false;

        goto cleanup;
    }

    else if (type == SECTION)
    {
        // Close previous section if open
        if (style.flags.section_open && (format == NULL || style.flags.content_printed))
        {
            printf("%s%s", section_color, style.chars.boxes[BOX_NORMAL][BOX_BOTTOM_LEFT]);
            for (size_t i = 0; i < style.total_width - 2; i++)
            {
                printf("%s", style.chars.boxes[BOX_NORMAL][BOX_HORIZONTAL]);
            }

            printf("%s%s\n", style.chars.boxes[BOX_NORMAL][BOX_BOTTOM_RIGHT], reset_code);

            style.flags.section_open = false;
            style.flags.content_printed = false;

            if (format == NULL)
            {
                goto cleanup;
            }
        }

        // Open new section
        if (format)
        {
            size_t dash_count = (style.total_width - 2 - buflen - 2) / 2;
            const size_t remaining = style.total_width - 2 - dash_count - buflen - 2;

            printf("%s%s", c_color, style.chars.boxes[BOX_NORMAL][BOX_TOP_LEFT]);

            if (!buffer[0])
            {
                dash_count += 2;
            }

            for (size_t i = 0; i < dash_count; i++)
            {
                printf("%s", style.chars.boxes[BOX_NORMAL][BOX_HORIZONTAL]);
            }

            if (buffer[0])
            {
                printf(" %s ", buffer);
            }

            for (size_t i = 0; i < remaining; i++)
            {
                printf("%s", style.chars.boxes[BOX_NORMAL][BOX_HORIZONTAL]);
            }

            printf("%s%s\n", style.chars.boxes[BOX_NORMAL][BOX_TOP_RIGHT], reset_code);

            style.flags.section_open = true;
            style.flags.content_printed = false;
        }

        goto cleanup;
    }

    else if (type == PROGRESS)
    {
        const int percent = margs.percent < 0 ? 0 : (margs.percent > 100 ? 100 : margs.percent);
        const int percent_width = percent < 10 ? 1 : (percent < 100 ? 2 : 3);
        const int metadata_width = 2 + 1 + percent_width + 1 + 1;
        const size_t bar_width = available - buflen - metadata_width - 1;
        const size_t filled_width = bar_width * percent / 100;
        const size_t empty_width = bar_width - filled_width;

        if (percent % 2)
        {
            c_color = style.chars.codes[COLOR_CYAN];
        }

        if (style.flags.section_open)
        {
            printf("%s%s", style.chars.ansi_escape_start, style.chars.ansi_carriage_return);
        }

        printf("%s%s%s ", section_color, box_vertical, c_color);

        printf("%s %s [", c_icon, buffer);

        for (size_t i = 0; i < filled_width; i++)
        {
            printf("%s", style.chars.progress_filled_char);
        }

        for (size_t i = 0; i < empty_width; i++)
        {
            printf("%s", style.chars.progress_empty_char);
        }

        printf("] %*d%%%s %s%s", percent_width, percent, section_color, box_vertical, reset_code);

        if (percent == 100)
        {
            printf("\n");
        }

        fflush(stdout);
        style.flags.content_printed = true;

        goto cleanup;
    }

    else if (type == CHOICE)
    {
        char* const* choices = margs.choices;
        int choice_count = 0;
        int selected = 0;

        while (choices[++choice_count])
            ;

        for (int i = 0; i < choice_count; i++)
        {
            if (simple_format)
            {
                printf("%d: %s\n", i + 1, choices[i]);
            }

            else
            {
                const size_t label_len = snprintf(NULL, 0, "%d: %s", i + 1, choices[i]);
                const size_t padding = label_len < available ? available - label_len + 2 : 0;

                printf("%s%s%s %d: %s%*s%s%s%s\n",
                       section_color,
                       box_vertical,
                       c_color,
                       i + 1,
                       choices[i],
                       (int)padding,
                       "",
                       section_color,
                       box_vertical,
                       reset_code);
            }
        }

        char input_buffer[TERMINAL_WIDTH] = { 0 };
        const char* w_msg = "Invalid input! Please enter a number between";
        const char* w_color = style.chars.codes[style.map[WARNING].color];
        const char* w_icon = style.chars.icons[ICON_WARNING];

        do
        {
            if (simple_format)
            {
                printf("%s (%d-%d): ", buffer, 1, choice_count);
            }

            else
            {
                printf("%s%s%s %s %s (1-%d): ",
                       section_color,
                       box_vertical,
                       c_color,
                       c_icon,
                       buffer,
                       choice_count);
            }

            fflush(stdout);
            terminal_read_input(input_buffer, sizeof(input_buffer));
            selected = atoi(input_buffer);

            if (!simple_format)
            {
                const size_t p_len = snprintf(NULL,
                                              0,
                                              "%s (1-%d): %s",
                                              buffer,
                                              choice_count,
                                              input_buffer);

                const size_t p_padding = p_len < available ? available - p_len : 0;

                printf("%*s%s%s%s\n", (int)p_padding, "", section_color, box_vertical, reset_code);
            }

            else
            {
                printf("\n");
            }

            if (selected >= 1 && selected <= choice_count)
            {
                style.flags.content_printed = true;
                va_end(args);
                return selected - 1;
            }

            if (simple_format)
            {
                printf("%s %d and %d.\n", w_msg, 1, choice_count);
            }

            else
            {
                const int w_len = snprintf(NULL, 0, "%s %s 1 and %d.", w_icon, w_msg, choice_count);

                const size_t w_padding = available > (size_t)w_len ? available - w_len + 2 : 0;

                printf("%s%s%s %s %s %d and %d.%*s%s%s%s\n",
                       section_color,
                       box_vertical,
                       w_color,
                       w_icon,
                       w_msg,
                       1,
                       choice_count,
                       (int)w_padding,
                       "",
                       section_color,
                       box_vertical,
                       reset_code);
            }
        } while (true);
    }

    else if (type == ALIAS)
    {
        char** const* alias_collections = margs.aliases;
        int collection_count = 0;
        int selected = -1;

        while (alias_collections[++collection_count])
            ;

        char input_buffer[TERMINAL_WIDTH] = { 0 };
        const char* w_msg = "Invalid input! Please enter a valid option.";
        const char* w_color = style.chars.codes[style.map[WARNING].color];
        const char* w_icon = style.chars.icons[ICON_WARNING];

        do
        {
            if (simple_format)
            {
                printf("%s: ", buffer);
            }

            else
            {
                printf("%s%s%s %s %s: ", section_color, box_vertical, c_color, c_icon, buffer);
            }

            fflush(stdout);
            terminal_read_input(input_buffer, sizeof(input_buffer));

            if (!simple_format)
            {
                const size_t p_len = snprintf(NULL, 0, "%s: %s", buffer, input_buffer);
                const size_t p_padding = p_len < available ? available - p_len : 0;

                printf("%*s%s%s%s\n", (int)p_padding, "", section_color, box_vertical, reset_code);
            }

            else
            {
                printf("\n");
            }

            bool found = false;
            for (int i = 0; i < collection_count && !found; i++)
            {
                char* const* aliases = alias_collections[i];
                for (int j = 0; aliases[j] != NULL; j++)
                {
                    if (strcasecmp(input_buffer, aliases[j]) == 0)
                    {
                        selected = i;
                        found = true;
                        break;
                    }
                }
            }

            if (found)
            {
                style.flags.content_printed = true;
                va_end(args);
                return selected;
            }

            if (simple_format)
            {
                printf("%s\n", w_msg);
            }

            else
            {
                const int e_len = snprintf(NULL, 0, "%s %s", w_icon, w_msg);

                const size_t e_padding = available > (size_t)e_len ? available - e_len + 2 : 0;

                printf("%s%s%s %s %s%*s%s%s%s\n",
                       section_color,
                       box_vertical,
                       w_color,
                       w_icon,
                       w_msg,
                       (int)e_padding,
                       "",
                       section_color,
                       box_vertical,
                       reset_code);
            }
        } while (true);
    }

    else if (type == PROMPT)
    {
        input_t input = margs.input;

        if (simple_format)
        {
            printf("%s: ", buffer);
        }

        else
        {
            printf("%s%s%s %s %s: ", section_color, box_vertical, c_color, c_icon, buffer);
        }

        fflush(stdout);
        terminal_read_input(input.ret, input.rsiz);

        if (!simple_format)
        {
            const size_t p_len = snprintf(NULL, 0, "%s: %s", buffer, input.ret);
            const size_t p_padding = p_len < available ? available - p_len : 0;

            printf("%*s%s%s%s\n", (int)p_padding, "", section_color, box_vertical, reset_code);
        }

        else
        {
            printf("\n");
        }

        style.flags.content_printed = true;

        goto cleanup;
    }

    else
    {
        const size_t padding = available > (size_t)buflen ? available - buflen : 0;

        if (simple_format)
        {
            printf("%s\n", buffer);
        }

        else
        {
            if (margs.loc != FIRST && icon_type != ICON_NONE)
            {
                c_icon = style.chars.boxes[BOX_NORMAL][margs.loc];
            }

            printf("%s%s%s ", section_color, box_vertical, c_color);

            if (icon_type != ICON_NONE)
            {
                printf("%s ", c_icon);
            }

            printf("%s", buffer);

            for (size_t i = 0; i < padding; i++)
            {
                printf(" ");
            }

            printf("%s%s%s\n", section_color, box_vertical, reset_code);
        }

        style.flags.content_printed = true;
    }

cleanup:
    va_end(args);
    return 0;
}

DESTRUCTOR static void
print_end_section()
{
    if (style.flags.section_open)
    {
        print(SECTION, MSG_NONE, NULL);
    }
}

#undef TERMINAL_WIDTH

#endif // PRINT_H