#ifndef PRINT_H
#define PRINT_H

#include "user.h"
#include "common.h"

#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_BOLD    "\x1b[1m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_GRAY    "\x1b[90m"

#define ICON_INFO     "•"
#define ICON_SUCCESS  "✓"
#define ICON_WARNING  "!"
#define ICON_ERROR    "✗"
#define ICON_CLOCK    "⧗"
#define ICON_DNA      "◇"
#define ICON_GEAR     "⚙"
#define ICON_DOT      "·"

#define BOX_TOP_LEFT     "┌"
#define BOX_TOP_RIGHT    "┐"
#define BOX_BOTTOM_LEFT  "└"
#define BOX_BOTTOM_RIGHT "┘"
#define BOX_HORIZONTAL   "─"
#define BOX_VERTICAL     "│"
#define BOX_TEE_RIGHT    "├"
#define BOX_TEE_LEFT     "┤"
#define BOX_TEE_DOWN     "┬"
#define BOX_TEE_UP       "┴"
#define BOX_CROSS        "┼"

#define FANCY_TOP_LEFT     "╔"
#define FANCY_TOP_RIGHT    "╗"
#define FANCY_BOTTOM_LEFT  "╚"
#define FANCY_BOTTOM_RIGHT "╝"
#define FANCY_HORIZONTAL   "═"
#define FANCY_VERTICAL     "║"

#define OUTPUT_WIDTH 70

typedef struct {
    int verbose;
    int quiet;
    int section_open;
    int content_printed;
    const char* box_color;
} MessageConfig;

static MessageConfig message_config = {0, 0, 0, 0, ANSI_COLOR_BLUE};

INLINE void init_colors(void) {
    message_config.box_color = ANSI_COLOR_BLUE;
}

INLINE void init_print_messages(int verbose, int quiet) {
    message_config.verbose = verbose;
    message_config.quiet = quiet;
    message_config.section_open = 0;
    message_config.content_printed = 0;
}

INLINE void print_newline(void) {
    if (!message_config.quiet) {
        printf("\n");
    }
}

INLINE void apply_box_color(void) {
    printf("%s%s", ANSI_COLOR_BOLD, message_config.box_color);
}

INLINE void reset_color(void) {
    printf("%s", ANSI_COLOR_RESET);
}

INLINE void print_box_line(const char* left, const char* mid, const char* right, int width, const char* title) {
    if (message_config.quiet) return;
    
    apply_box_color();
    printf("%s", left);
    
    if (title && strlen(title) > 0) {
        int title_len = strlen(title);
        int remaining_width = width - 2;
        int padding_left = (remaining_width - title_len - 2) / 2;
        int padding_right = remaining_width - title_len - 2 - padding_left;
        for (int i = 0; i < padding_left; i++) printf("%s", mid);
        printf(" %s ", title);
        for (int i = 0; i < padding_right; i++) printf("%s", mid);
    } else {
        for (int i = 0; i < width - 2; i++) printf("%s", mid);
    }
    
    printf("%s", right);
    
    reset_color();
    printf("\n");
}

INLINE void print_step_header_start(const char* title) {
    if (message_config.quiet) return;
    
    if (message_config.section_open) {
        if (message_config.content_printed) {
            print_box_line(BOX_BOTTOM_LEFT, BOX_HORIZONTAL, BOX_BOTTOM_RIGHT, OUTPUT_WIDTH, NULL);
        } else {
            // If no content was printed, just overwrite the header
            printf("\r");
        }
    }
    
    print_box_line(BOX_TOP_LEFT, BOX_HORIZONTAL, BOX_TOP_RIGHT, OUTPUT_WIDTH, title);
    message_config.section_open = 1;
    message_config.content_printed = 0;
}

INLINE void print_step_header_end(void) {
    if (message_config.quiet || !message_config.section_open) return;
    print_box_line(BOX_BOTTOM_LEFT, BOX_HORIZONTAL, BOX_BOTTOM_RIGHT, OUTPUT_WIDTH, NULL);
    message_config.section_open = 0;
    message_config.content_printed = 0;
}

INLINE void sanitize_message(char* buffer, size_t buffer_size, const char* format, va_list args) {
    vsnprintf(buffer, buffer_size, format, args);
    
    for (size_t i = 0; i < strlen(buffer); i++) {
        if (buffer[i] < 32 || buffer[i] > 126) {
            buffer[i] = ' ';
        }
    }
    
    size_t len = strlen(buffer);
    while (len > 0 && isspace(buffer[len - 1])) {
        buffer[--len] = '\0';
    }
}

INLINE void print_formatted_message(const char* icon, const char* color, const char* format, va_list args) {
    if (message_config.quiet) return;
    
    if (!message_config.section_open) {
        print_step_header_start("Information");
    }
    
    char buffer[OUTPUT_WIDTH * 2];
    sanitize_message(buffer, sizeof(buffer), format, args);
    
    size_t content_len = strlen(buffer);
    size_t icon_space = 2;
    size_t total_len = icon_space + content_len;
    int padding = (int)(OUTPUT_WIDTH - 2 - total_len - 1);
    
    // Handle overflow by truncating the message
    if (padding < 0) {
        content_len += padding;
        if (content_len > 0) {
            buffer[content_len] = '\0';
        }
        padding = 0;
    }
    
    apply_box_color();
    printf("%s", BOX_VERTICAL);
    reset_color();
    printf(" %s%s %s%s", color, icon, buffer, ANSI_COLOR_RESET);
    
    for (int i = 0; i < padding; i++) printf(" ");
    
    apply_box_color();
    printf("%s", BOX_VERTICAL);
    reset_color();
    
    printf("\n");
    message_config.content_printed = 1;
}

INLINE void print_info(const char* format, ...) {
    if (message_config.quiet) return;
    va_list args;
    va_start(args, format);
    print_formatted_message(ICON_INFO, ANSI_COLOR_BLUE, format, args);
    va_end(args);
}

INLINE void print_success(const char* format, ...) {
    if (message_config.quiet) return;
    va_list args;
    va_start(args, format);
    print_formatted_message(ICON_SUCCESS, ANSI_COLOR_GREEN, format, args);
    va_end(args);
}

INLINE void print_warning(const char* format, ...) {
    va_list args;
    va_start(args, format);
    print_formatted_message(ICON_WARNING, ANSI_COLOR_YELLOW, format, args);
    va_end(args);
}

INLINE void print_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    print_formatted_message(ICON_ERROR, ANSI_COLOR_RED ANSI_COLOR_BOLD, format, args);
    va_end(args);
}

INLINE void print_timing(const char* format, ...) {
    if (message_config.quiet) return;
    va_list args;
    va_start(args, format);
    print_formatted_message(ICON_CLOCK, ANSI_COLOR_CYAN, format, args);
    va_end(args);
}

INLINE void print_dna(const char* format, ...) {
    if (message_config.quiet) return;
    va_list args;
    va_start(args, format);
    print_formatted_message(ICON_DNA, ANSI_COLOR_MAGENTA, format, args);
    va_end(args);
}

INLINE void print_config(const char* format, ...) {
    if (message_config.quiet) return;
    va_list args;
    va_start(args, format);
    print_formatted_message(ICON_GEAR, ANSI_COLOR_YELLOW, format, args);
    va_end(args);
}

INLINE void print_verbose(const char* format, ...) {
    if (message_config.quiet || !message_config.verbose) return;
    va_list args;
    va_start(args, format);
    print_formatted_message(ICON_DOT, ANSI_COLOR_GRAY, format, args);
    va_end(args);
}

INLINE void print_progress_bar(double percentage, size_t width, const char* prefix) {
    if (message_config.quiet) return;
    
    if (!message_config.section_open) {
        print_step_header_start("Progress");
    }
    
    size_t prefix_len = strlen(prefix);
    size_t reserved_space = 2 + 2 + 2 + 6;
    int max_bar_width = (int)(OUTPUT_WIDTH - prefix_len - reserved_space);
    if ((int)width > max_bar_width) width = max_bar_width;
    size_t filled_width = (size_t)(percentage * width);
    if (filled_width > width) filled_width = width;
    
    apply_box_color();
    printf("\r%s", BOX_VERTICAL);
    reset_color();
    
    printf(" %s [", prefix);
    
    printf("%s", ANSI_COLOR_GREEN);
    for (size_t i = 0; i < filled_width; ++i) printf("■");
    printf("%s", ANSI_COLOR_RESET);
    
    for (size_t i = filled_width; i < width; ++i) printf(ICON_DOT);
    
    printf("] %3d%%", (int)(percentage * 100));
    
    int content_len = prefix_len + width + 7;
    int padding = OUTPUT_WIDTH - 3 - content_len - 1;
    
    for (int i = 0; i < padding; i++) printf(" ");
    
    apply_box_color();
    printf("%s", BOX_VERTICAL);
    reset_color();
    
    fflush(stdout);
    message_config.content_printed = 1;
}

INLINE void print_header(const char* title, char fill_char) {
    if (message_config.quiet) return;
    
    printf("%s%s", ANSI_COLOR_BOLD, ANSI_COLOR_BLUE);
    
    printf("%s", FANCY_TOP_LEFT);
    for (int i = 0; i < OUTPUT_WIDTH - 2; i++) printf("%s", FANCY_HORIZONTAL);

    printf("%s\n", FANCY_TOP_RIGHT);
    printf("%s", FANCY_VERTICAL);
    
    size_t title_len = strlen(title);
    size_t padding = (OUTPUT_WIDTH - 2 - title_len) / 2;
    
    for (int i = 0; i < padding; i++) printf(" ");

    printf("%s", title);

    for (int i = 0; i < OUTPUT_WIDTH - 2 - title_len - padding; i++) printf(" ");
    
    printf("%s\n", FANCY_VERTICAL);
    printf("%s", FANCY_BOTTOM_LEFT);
    for (int i = 0; i < OUTPUT_WIDTH - 2; i++) printf("%s", FANCY_HORIZONTAL);
    printf("%s", FANCY_BOTTOM_RIGHT);
    
    printf("%s\n", ANSI_COLOR_RESET);
}

INLINE void print_step_header(const char* title) {
    print_step_header_start(title);
}

INLINE void print_config_item(const char* item, const char* value, const char* prefix) {
    if (message_config.quiet) return;
    
    static int first_config_item = 1;
    
    if (first_config_item) {
        char format[OUTPUT_WIDTH];
        snprintf(format, sizeof(format), "%s:  %s", item, value);
        print_config("%s", format);
        first_config_item = 0;
    } else {
        char buffer[OUTPUT_WIDTH * 2];
        snprintf(buffer, sizeof(buffer), "%s %s: %s", prefix, item, value);
        if (!message_config.section_open) {
            print_step_header_start("Configuration");
        }

        size_t content_len = strlen(buffer);
        int padding = (int)(OUTPUT_WIDTH - content_len - 1);
        if (padding < 0) padding = 0;
        
        apply_box_color();
        printf("%s", BOX_VERTICAL);
        reset_color();
        printf(" %s%s", ANSI_COLOR_YELLOW, buffer);
        
        for (int i = 0; i < padding; i++) printf(" ");
        
        apply_box_color();
        printf("%s", BOX_VERTICAL);
        reset_color();
        
        printf("\n");
        message_config.content_printed = 1;
    }
}

INLINE void reset_config_counter(void) {
    static int first_config_item = 1;
    first_config_item = 1;
}

#endif // PRINT_H