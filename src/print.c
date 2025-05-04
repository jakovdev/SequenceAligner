#include "print.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#if defined(__STDC__) && !defined(__STDC_VERSION__)
#ifdef _WIN32
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#else
#pragma message("Warning: Using unsafe sprintf/vsprintf instead of snprintf/vsnprintf")
#define snprintf sprintf
#define vsnprintf(a, b, c, d) vsprintf(a, c, d)
#endif
#endif

#define TERMINAL_WIDTH 80

#ifdef _WIN32
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

static int
terminal_environment(void)
{
    static int is_terminal = -1;
    if (is_terminal == -1)
    {
#ifdef _WIN32
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD dwMode = 0;
        is_terminal = (hStdout != INVALID_HANDLE_VALUE && GetConsoleMode(hStdout, &dwMode));
#else
        is_terminal = isatty(STDOUT_FILENO);
#endif
    }

    return is_terminal;
}

static void
terminal_init(void)
{
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE)
    {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode))
        {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }

#endif
}

static void
terminal_mode_raw(void)
{
#ifdef _WIN32
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(hStdin, &mode);
    SetConsoleMode(hStdin, mode & ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT));
#else
    struct termios term;
    tcgetattr(STDIN_FILENO, &term);
    term.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &term);
#endif
}

static void
terminal_mode_restore(void)
{
#ifdef _WIN32
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(hStdin, &mode);
    SetConsoleMode(hStdin, mode | (ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT));
#else
    struct termios term;
    tcgetattr(STDIN_FILENO, &term);
    term.c_lflag |= (ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &term);
#endif
}

static void
terminal_read_input(char* input_buffer, int input_buffer_size)
{
    int input_character_index = 0;
    int input_character;

    fflush(stdout);

    terminal_mode_raw();

    while (1)
    {
        input_character = getchar();

        if (input_character == '\n' || input_character == '\r')
        {
            break;
        }

        if (input_character == '\x7F' || input_character == '\b')
        {
            if (input_character_index > 0)
            {
                input_buffer[--input_character_index] = '\0';
                printf("\b \b");
                fflush(stdout);
            }

            continue;
        }

        if (input_character_index < input_buffer_size - 1)
        {
            input_buffer[input_character_index++] = input_character;
            input_buffer[input_character_index] = '\0';
            printf("%c", input_character);
            fflush(stdout);
        }
    }

    terminal_mode_restore();
}

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
        unsigned is_init : 1;
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

void
print_verbose_flip()
{
    style.flags.verbose = !style.flags.verbose;
}

void
print_quiet_flip()
{
    style.flags.quiet = !style.flags.quiet;
}

static void
print_context_init()
{
    terminal_init();
    if (!terminal_environment())
    {
        style.chars.ansi_carriage_return = "\n";
    }

    style.flags.is_init = 1;
}

#ifdef __GNUC__
#define DESTRUCTOR __attribute__((destructor))
#else
#define DESTRUCTOR
#endif

#ifdef __cplusplus
#define P_RESTRICT __restrict
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define P_RESTRICT restrict
#else
#define P_RESTRICT
#endif

DESTRUCTOR static void
print_end_section()
{
    if (style.flags.section_open)
    {
        print(SECTION, MSG_NONE, NULL);
    }
}

int
print(message_t type, MSG_ARG margs, const char* P_RESTRICT format, ...)
{
    if (!style.flags.is_init)
    {
        print_context_init();
#ifdef _MSC_VER
        atexit(print_end_section)
#endif
    }

    const int is_required = style.map[type].requirement == REQUIRED;
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

    const int simple_format = style.flags.quiet && is_required;
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

        /* Top border */
        printf("%s%s", c_color, style.chars.boxes[BOX_FANCY][BOX_TOP_LEFT]);
        size_t i;
        for (i = 0; i < style.total_width - 2; i++)
        {
            printf("%s", style.chars.boxes[BOX_FANCY][BOX_HORIZONTAL]);
        }

        printf("%s%s\n", style.chars.boxes[BOX_FANCY][BOX_TOP_RIGHT], reset_code);

        /* Content with centering */
        const size_t left_padding = (style.total_width - 2 - buflen) / 2;
        const size_t right_padding = style.total_width - 2 - buflen - left_padding;

        printf("%s%s", c_color, style.chars.boxes[BOX_FANCY][BOX_VERTICAL]);
        printf("%*s%s%*s", (int)left_padding, "", buffer, (int)right_padding, "");
        printf("%s%s\n", style.chars.boxes[BOX_FANCY][BOX_VERTICAL], reset_code);

        /* Bottom border */
        printf("%s%s", c_color, style.chars.boxes[BOX_FANCY][BOX_BOTTOM_LEFT]);
        for (i = 0; i < style.total_width - 2; i++)
        {
            printf("%s", style.chars.boxes[BOX_FANCY][BOX_HORIZONTAL]);
        }

        printf("%s%s\n", style.chars.boxes[BOX_FANCY][BOX_BOTTOM_RIGHT], reset_code);

        style.flags.section_open = 0;

        goto cleanup;
    }

    else if (type == SECTION)
    {
        /* Close previous section if open */
        if (style.flags.section_open && (format == NULL || style.flags.content_printed))
        {
            printf("%s%s", section_color, style.chars.boxes[BOX_NORMAL][BOX_BOTTOM_LEFT]);
            size_t i;
            for (i = 0; i < style.total_width - 2; i++)
            {
                printf("%s", style.chars.boxes[BOX_NORMAL][BOX_HORIZONTAL]);
            }

            printf("%s%s\n", style.chars.boxes[BOX_NORMAL][BOX_BOTTOM_RIGHT], reset_code);

            style.flags.section_open = 0;
            style.flags.content_printed = 0;

            if (format == NULL)
            {
                goto cleanup;
            }
        }

        /* Open new section */
        if (format)
        {
            size_t dash_count = (style.total_width - 2 - buflen - 2) / 2;
            const size_t remaining = style.total_width - 2 - dash_count - buflen - 2;

            printf("%s%s", c_color, style.chars.boxes[BOX_NORMAL][BOX_TOP_LEFT]);

            if (!buffer[0])
            {
                dash_count += 2;
            }

            size_t i;
            for (i = 0; i < dash_count; i++)
            {
                printf("%s", style.chars.boxes[BOX_NORMAL][BOX_HORIZONTAL]);
            }

            if (buffer[0])
            {
                printf(" %s ", buffer);
            }

            for (i = 0; i < remaining; i++)
            {
                printf("%s", style.chars.boxes[BOX_NORMAL][BOX_HORIZONTAL]);
            }

            printf("%s%s\n", style.chars.boxes[BOX_NORMAL][BOX_TOP_RIGHT], reset_code);

            style.flags.section_open = 1;
            style.flags.content_printed = 0;
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

        size_t i;
        for (i = 0; i < filled_width; i++)
        {
            printf("%s", style.chars.progress_filled_char);
        }

        for (i = 0; i < empty_width; i++)
        {
            printf("%s", style.chars.progress_empty_char);
        }

        printf("] %*d%%%s %s%s", percent_width, percent, section_color, box_vertical, reset_code);

        if (percent == 100)
        {
            printf("\n");
        }

        fflush(stdout);
        style.flags.content_printed = 1;

        goto cleanup;
    }

    else if (type == CHOICE)
    {
        char* const* choices = margs.choices;
        int choice_count = 0;
        int selected = 0;

        while (choices[++choice_count])
            ;

        int i;
        for (i = 0; i < choice_count; i++)
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
                style.flags.content_printed = 1;
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
        } while (1);
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

            int found = 0;
            int i, j;
            for (i = 0; i < collection_count && !found; i++)
            {
                char* const* aliases = alias_collections[i];
                for (j = 0; aliases[j] != NULL; j++)
                {
                    if (strcasecmp(input_buffer, aliases[j]) == 0)
                    {
                        selected = i;
                        found = 1;
                        break;
                    }
                }
            }

            if (found)
            {
                style.flags.content_printed = 1;
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
        } while (1);
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

        style.flags.content_printed = 1;

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

            size_t i;
            for (i = 0; i < padding; i++)
            {
                printf(" ");
            }

            printf("%s%s%s\n", section_color, box_vertical, reset_code);
        }

        style.flags.content_printed = 1;
    }

cleanup:
    va_end(args);
    return 0;
}