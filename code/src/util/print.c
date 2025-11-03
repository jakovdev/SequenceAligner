#include "util/print.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef ERROR
#undef ERROR
#endif

#ifdef OPTIONAL
#undef OPTIONAL
#endif

#ifdef REQUIRED
#undef REQUIRED
#endif
#define MAX max
#define MIN min
#define fputc_unlocked _fputc_nolock
#define fputs_unlocked _fputs_nolock
#define fwrite_unlocked _fwrite_nolock
#else
#include <termios.h>
#include <unistd.h>
#include <sys/param.h>
#endif

#define CLAMP(v, min, max) (MIN(MAX((v), (min)), (max)))

#ifndef TERMINAL_WIDTH
#define TERMINAL_WIDTH 80
#endif

static int terminal_environment(void)
{
	static int is_terminal = -1;
	if (is_terminal == -1) {
#ifdef _WIN32
		HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
		DWORD dwMode = 0;
		is_terminal = (hStdout != INVALID_HANDLE_VALUE &&
			       GetConsoleMode(hStdout, &dwMode));
#else
		is_terminal = isatty(STDOUT_FILENO);
#endif
	}

	return is_terminal;
}

static void terminal_init(void)
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
	SetConsoleCP(CP_UTF8);

	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hOut != INVALID_HANDLE_VALUE) {
		DWORD dwMode = 0;
		if (GetConsoleMode(hOut, &dwMode)) {
			dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
			SetConsoleMode(hOut, dwMode);
		}
	}

#endif
}

static void terminal_mode_raw(void)
{
#ifdef _WIN32
	HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
	DWORD mode, mask = ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT;
	if (GetConsoleMode(hStdin, &mode))
		SetConsoleMode(hStdin, mode & ~mask);

#else
	struct termios term;
	tcgetattr(STDIN_FILENO, &term);
	term.c_lflag &= ~((tcflag_t)(ICANON | ECHO));
	tcsetattr(STDIN_FILENO, TCSANOW, &term);
#endif
}

static void terminal_mode_restore(void)
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

enum {
	M_INFO = 1,
	M_VERBOSE,
	M_WARNING,
	M_ERROR,
	M_CHOICE,
	M_PROMPT,
	M_PROGRESS,
	M_HEADER,
	M_SECTION,
	M_TYPES
};

typedef enum {
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

typedef enum {
	ICON_NONE,
	ICON_INFO,
	ICON_DOT,
	ICON_WARNING,
	ICON_ERROR,
	ICON_ARROW,
	ICON_TYPE_COUNT
} icon_t;

typedef enum { OPTIONAL, REQUIRED } requirement_t;

enum {
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

enum { BOX_NORMAL, BOX_FANCY, BOX_TYPE_COUNT };

static struct {
	const struct {
		color_t color;
		icon_t icon;
		requirement_t requirement;
	} map[M_TYPES];
	struct {
		unsigned verbose : 1;
		unsigned quiet : 1;
		unsigned nodetail : 1;
		unsigned in_section : 1;
		unsigned content_printed : 1;
		unsigned is_init : 1;
	} flags;
	const char *codes[COLOR_TYPE_COUNT];
	const char *icons[ICON_TYPE_COUNT];
	const char *boxes[BOX_TYPE_COUNT][BOX_CHAR_COUNT];
	const char *progress_filled_char;
	const char *progress_empty_char;
	const char *ansi_escape_start;
	const char *ansi_carriage_return;
	FILE *in;
	FILE *out;
	FILE *err;
	size_t width;
	char err_ctx[TERMINAL_WIDTH];

} p = {
    .map = {
        [0]          = { COLOR_RESET,       ICON_NONE,    OPTIONAL },
        [M_INFO]     = { COLOR_BLUE,        ICON_INFO,    OPTIONAL },
        [M_VERBOSE]  = { COLOR_GRAY,        ICON_DOT,     REQUIRED },
        [M_WARNING]  = { COLOR_YELLOW,      ICON_WARNING, REQUIRED },
        [M_ERROR]    = { COLOR_RED,         ICON_ERROR,   REQUIRED },
        [M_CHOICE]   = { COLOR_BLUE,        ICON_INFO,    REQUIRED },
        [M_PROMPT]   = { COLOR_BLUE,        ICON_INFO,    REQUIRED },
        [M_PROGRESS] = { COLOR_BRIGHT_CYAN, ICON_ARROW,   OPTIONAL },
        [M_HEADER]   = { COLOR_BRIGHT_CYAN, ICON_NONE,    OPTIONAL },
        [M_SECTION]  = { COLOR_BLUE,        ICON_NONE,    OPTIONAL },
    },
#define CODESIZ(c) (c == COLOR_RESET ? 4 : 5)
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
        [ICON_DOT]     = "·",
        [ICON_WARNING] = "!",
        [ICON_ERROR]   = "✗",
        [ICON_ARROW]   = "▶",
    },
#define BOXSIZ (sizeof("┌") - 1)
    .boxes = {
        { "┌", "├", "└", "┐", "─", "│", "┤", "┘" },
        { "╔", "╠", "╚", "╗", "═", "║", "╣", "╝" },
    },
    .progress_filled_char = "■",
    .progress_empty_char = "·",
    .ansi_escape_start = "\x1b",
    .ansi_carriage_return = "\r",
    .width = TERMINAL_WIDTH,
};

void print_verbose_flip(void)
{
	p.flags.verbose = !p.flags.verbose;
}

void print_quiet_flip(void)
{
	p.flags.quiet = !p.flags.quiet;
}

void print_detail_flip(void)
{
	p.flags.nodetail = !p.flags.nodetail;
}

void print_error_context(const char *context)
{
	if (!context) {
		p.err_ctx[0] = '\0';
		return;
	}

	snprintf(p.err_ctx, sizeof(p.err_ctx), "%s | ", context);
}

void print_streams(FILE *in, FILE *out, FILE *err)
{
	p.in = in;
	p.out = out;
	p.err = err;
}

static void print_init(void)
{
	terminal_init();
	if (!terminal_environment())
		p.ansi_carriage_return = "\n";

	print_streams(stdin, stdout, stderr);
	p.flags.is_init = 1;
}

static void terminal_read_input(char *input_buffer, size_t input_buffer_size)
{
	size_t input_char_index = 0;
	int input_char;

	fflush(p.out);
	terminal_mode_raw();

	while ((input_char = fgetc(p.in)) != EOF) {
		if (input_char == '\n' || input_char == '\r')
			break;

		if (input_char == '\x7F' || input_char == '\b') {
			if (input_char_index > 0) {
				input_buffer[--input_char_index] = '\0';
				fputs("\b \b", p.out);
				fflush(p.out);
			}

			continue;
		}

		if (input_char_index + 1 < input_buffer_size) {
			input_buffer[input_char_index++] = (char)input_char;
			input_buffer[input_char_index] = '\0';
			fputc(input_char, p.out);
			fflush(p.out);
		}
	}

	terminal_mode_restore();
}

int print_yN(const char *P_RESTRICT prompt)
{
	char result[2] = { 0 };
	print(M_UINS(result), PROMPT "%s [y/N]: ", prompt);
	if (result[0] == 'y' || result[0] == 'Y')
		return PRINT_USER_YES;
	else
		return PRINT_USER_NO;
}

int print_Yn(const char *P_RESTRICT prompt)
{
	char result[2] = { 0 };
	print(M_UINS(result), PROMPT "%s [Y/n]: ", prompt);
	if (result[0] == 'n' || result[0] == 'N')
		return PRINT_USER_NO;
	else
		return PRINT_USER_YES;
}

int print_yn(const char *P_RESTRICT prompt)
{
	char result[2] = { 0 };
repeat:
	print(M_UINS(result), PROMPT "%s [y/n]: ", prompt);
	if (result[0] == 'y' || result[0] == 'Y')
		return PRINT_USER_YES;
	else if (result[0] == 'n' || result[0] == 'N')
		return PRINT_USER_NO;

	goto repeat;
}

DESTRUCTOR static void print_section_end(void)
{
	if (p.flags.in_section)
		print(M_NONE, NULL);
}

int print(M_ARG arg, const char *P_RESTRICT fmt, ...)
{
	if (!p.flags.is_init) {
		print_init();
#ifdef _MSC_VER
		atexit(print_end_section);
#endif
	}

	int type = 0;
	if (fmt && fmt[0] >= '0' && fmt[0] <= '9') {
		type = fmt[0] - '0';
		fmt++;
	}

	FILE *out = (type == M_ERROR) ? p.err : p.out;
#define ouputc(c) fputc_unlocked((c), out)
#define ouputs(s) fputs_unlocked((s), out)
#define oprintf(...) fprintf(out, __VA_ARGS__)
#define ouwrite(buf, size) fwrite_unlocked((buf), 1, (size), out)

	const int is_verbose = type == M_VERBOSE && !p.flags.verbose;
	const int is_required = p.map[type].requirement == REQUIRED;
	if ((p.flags.quiet && !is_required) || is_verbose)
		return PRINT_SKIPPED_BECAUSE_QUIET_OR_VERBOSE_NOT_ENABLED__SUCCESS;

	if (!p.flags.in_section && type != M_HEADER && type != M_SECTION)
		print(M_NONE, SECTION);

	static int last_percentage = -1;
	if (type == M_PROGRESS) {
		const int percent = CLAMP(arg.percent, 0, 100);
		if (percent == last_percentage)
			return PRINT_REPEAT_PROGRESS_PERCENT__SUCCESS;

		last_percentage = percent;

		if (last_percentage == 100)
			last_percentage = -1;
	} else if (last_percentage != -1 && p.flags.content_printed) {
		fputc('\n', out);
		last_percentage = -1;
	}

	const int simple = p.flags.nodetail || (p.flags.quiet && is_required);

#define PMAP(t) p.map[(t)]
#define PICO(t) PMAP(t).icon
#define PCOL(t) PMAP(t).color
#define ouwico(t) ouwrite(p.icons[PICO(t)], strlen(p.icons[PICO(t)]))
#define ouwcol(t) ouwrite(p.codes[PCOL(t)], CODESIZ(PCOL(t)))
#define ouwbox(t, part) ouwrite(p.boxes[(t)][part], BOXSIZ)

	const size_t box_chars = simple ? 0 : 1;
	const size_t ico_chars = (simple || PMAP(type).icon == ICON_NONE) ? 0 :
									    2;
	const size_t available = p.width - (2 * box_chars) - ico_chars - 1;

	char p_buf[BUFSIZ] = { 0 };
	int p_bufsiz = 0;

	if (fmt) {
		char fmt_copy[BUFSIZ];
		size_t fmt_i = 0;

		fmt_copy[0] = '\0';

		if (type == M_ERROR && p.err_ctx[0] != '\0') {
			size_t ctxlen =
				strnlen(p.err_ctx, sizeof(fmt_copy) - 1);
			if (ctxlen) {
				memcpy(fmt_copy + fmt_i, p.err_ctx, ctxlen);
				fmt_i += ctxlen;
			}
		}

		{
			const char *fmt_src = fmt ? fmt : "";
			size_t copy_len =
				strnlen(fmt_src, sizeof(fmt_copy) - 1 - fmt_i);
			if (copy_len)
				memcpy(fmt_copy + fmt_i, fmt_src, copy_len);
			fmt_copy[fmt_i + copy_len] = '\0';
		}

		va_list v_args;
		va_start(v_args, fmt);
		p_bufsiz = vsnprintf(p_buf, sizeof(p_buf), fmt_copy, v_args);
		va_end(v_args);

		if (p_bufsiz < 0) {
#if DEFINE_AS_1_TO_TURN_OFF_DEV_MESSAGES == 0
			print_error_context("_TO_DEV_");
			print(M_NONE, ERROR "Failed to format string");
#endif
			return PRINT_INVALID_FORMAT_ARGS__ERROR;
		}
	} else {
		/* No format string, used for section end */
		type = M_SECTION;
	}

	size_t p_buflen = (size_t)p_bufsiz;

	flockfile(out);

	if (type == M_HEADER) {
		if (!fmt)
			goto cleanup;

		if (simple) {
			ouputc('\n');
			ouwrite(p_buf, p_buflen);
			ouwrite("\n\n", 2);
			p.flags.in_section = 0;
			goto cleanup;
		}

		if (p.flags.in_section)
			print(M_NONE, NULL);

		/* Top border */
		ouwcol(type);
		ouwbox(BOX_FANCY, BOX_TOP_LEFT);

		size_t iwlrp;
		for (iwlrp = 0; iwlrp < p.width - 2; iwlrp++)
			ouwbox(BOX_FANCY, BOX_HORIZONTAL);

		ouwbox(BOX_FANCY, BOX_TOP_RIGHT);
		ouwcol(COLOR_RESET);
		ouputc('\n');

		/* Content with centering */
		const size_t l_pad = (p.width - 2 - p_buflen) / 2;
		const size_t r_pad = p.width - 2 - p_buflen - l_pad;

		ouwcol(type);
		ouwbox(BOX_FANCY, BOX_VERTICAL);
		for (iwlrp = 0; iwlrp < l_pad; iwlrp++)
			ouputc(' ');
		ouwrite(p_buf, p_buflen);
		for (iwlrp = 0; iwlrp < r_pad; iwlrp++)
			ouputc(' ');
		ouwbox(BOX_FANCY, BOX_VERTICAL);
		ouwcol(COLOR_RESET);
		ouputc('\n');

		/* Bottom border */
		ouwcol(type);
		ouwbox(BOX_FANCY, BOX_BOTTOM_LEFT);
		for (iwlrp = 0; iwlrp < p.width - 2; iwlrp++)
			ouwbox(BOX_FANCY, BOX_HORIZONTAL);

		ouwbox(BOX_FANCY, BOX_BOTTOM_RIGHT);
		ouwcol(COLOR_RESET);
		ouputc('\n');

		p.flags.in_section = 0;

		goto cleanup;
	} else if (type == M_SECTION) {
		/* Close previous section if open */
		if (p.flags.in_section && (!fmt || p.flags.content_printed)) {
			if (simple) {
				ouputc('\n');

				p.flags.in_section = 0;
				p.flags.content_printed = 0;
			} else {
				ouwcol(M_SECTION);
				ouwbox(BOX_NORMAL, BOX_BOTTOM_LEFT);

				size_t iw;
				for (iw = 0; iw < p.width - 2; iw++)
					ouwbox(BOX_NORMAL, BOX_HORIZONTAL);

				ouwbox(BOX_NORMAL, BOX_BOTTOM_RIGHT);
				ouwcol(COLOR_RESET);
				ouputc('\n');

				p.flags.in_section = 0;
				p.flags.content_printed = 0;
			}
		}

		if (!fmt)
			goto cleanup;

		/* Open new section */
		if (simple) {
			ouwrite(p_buf, p_buflen);
			ouputc('\n');

			p.flags.in_section = 1;
			p.flags.content_printed = 0;
			goto cleanup;
		}

		size_t l_dashes = (p.width - 2 - p_buflen - 2) / 2;
		size_t r_dashes = p.width - 2 - l_dashes - p_buflen - 2;

		ouwcol(M_SECTION);
		ouwbox(BOX_NORMAL, BOX_TOP_LEFT);

		if (!p_buf[0])
			l_dashes += 2;

		while (l_dashes--)
			ouwbox(BOX_NORMAL, BOX_HORIZONTAL);

		if (p_buf[0]) {
			ouputc(' ');
			ouwrite(p_buf, p_buflen);
			ouputc(' ');
		}

		while (r_dashes--)
			ouwbox(BOX_NORMAL, BOX_HORIZONTAL);

		ouwbox(BOX_NORMAL, BOX_TOP_RIGHT);
		ouwcol(COLOR_RESET);
		ouputc('\n');

		p.flags.in_section = 1;
		p.flags.content_printed = 0;
		goto cleanup;
	} else if (type == M_PROGRESS) {
		const int percent = CLAMP(arg.percent, 0, 100);
		const int percent_width =
			percent < 10 ? 1 : (percent < 100 ? 2 : 3);

		if (simple) {
			if (p.flags.in_section)
				ouputs(p.ansi_carriage_return);

			ouputs(p_buf);
			ouputc(' ');
			if (percent == 100) {
				ouwrite("100%\n", 5);
			} else {
				if (percent >= 10)
					ouputc('0' + percent / 10);
				ouputc('0' + percent % 10);
				ouputc('%');
			}

			fflush(out);
			p.flags.content_printed = 1;
			goto cleanup;
		}

		const size_t meta_width = (size_t)percent_width + 2 + 1 + 1 + 1;
		const size_t bar_width = available - p_buflen - meta_width - 1;
		const size_t filled_width = bar_width * (size_t)percent / 100;
		const size_t empty_width = bar_width - filled_width;

		if (p.flags.in_section) {
			ouputs(p.ansi_carriage_return);
		}

		ouwcol(M_SECTION);
		ouwbox(BOX_NORMAL, BOX_VERTICAL);
		if (percent % 2)
			ouwcol(COLOR_CYAN);
		else
			ouwcol(type);

		ouputc(' ');
		ouwico(type);
		ouputc(' ');
		ouwrite(p_buf, p_buflen);
		ouwrite(" [", 2);

		size_t ifew;
		for (ifew = 0; ifew < filled_width; ifew++)
			ouputs(p.progress_filled_char);

		for (ifew = 0; ifew < empty_width; ifew++)
			ouputs(p.progress_empty_char);

		ouwrite("] ", 2);
		if (percent == 100) {
			ouwrite("100% ", 5);
		} else {
			if (percent >= 10)
				ouputc('0' + percent / 10);
			ouputc('0' + percent % 10);
			ouwrite("% ", 2);
		}

		ouwcol(M_SECTION);
		ouwbox(BOX_NORMAL, BOX_VERTICAL);
		ouwcol(COLOR_RESET);
		if (percent == 100)
			ouputc('\n');

		fflush(out);
		p.flags.content_printed = 1;

		goto cleanup;
	} else if (type == M_CHOICE) {
		char **choices = arg.choice.choices;
		size_t c_count = arg.choice.n;

		if (c_count < 2) {
#if DEFINE_AS_1_TO_TURN_OFF_DEV_MESSAGES == 0
			print_error_context("_TO_DEV_");
			print(M_NONE, ERROR "Not enough choices (<2)");
#endif
			return PRINT_CHOICE_COLLECTION_SHOULD_CONTAIN_2_OR_MORE_CHOICES__ERROR;
		}

		choices[c_count] = NULL;

		size_t c;
		for (c = 0; c < c_count; c++) {
			if (simple) {
				oprintf("%zu: %s\n", c + 1, choices[c]);
			} else {
				const int label_chars = snprintf(
					NULL, 0, "%zu: %s", c + 1, choices[c]);
				const size_t label_len =
					(label_chars < 0) ? 0 :
							    (size_t)label_chars;
				const size_t padding =
					label_len < available ?
						available - label_len + 2 :
						0;

				ouwcol(M_SECTION);
				ouwbox(BOX_NORMAL, BOX_VERTICAL);
				ouwcol(type);
				oprintf(" %zu: %s%*s", c + 1, choices[c],
					(int)padding, "");
				ouwcol(M_SECTION);
				ouwbox(BOX_NORMAL, BOX_VERTICAL);
				ouwcol(COLOR_RESET);
				ouputc('\n');
			}
		}

		char i_buffer[TERMINAL_WIDTH] = { 0 };
		const char *w_msg = "Please enter a number between";
		const char *w_color = p.codes[p.map[M_WARNING].color];
		const char *w_icon = p.icons[ICON_WARNING];

		do {
			if (!simple) {
				ouwcol(M_SECTION);
				ouwbox(BOX_NORMAL, BOX_VERTICAL);
				ouwcol(type);
				ouputc(' ');
				ouwico(type);
				ouputc(' ');
			}

			ouwrite(p_buf, p_buflen);
			ouwrite(" (1-", 4);
			oprintf("%zu", c_count);
			ouwrite("): ", 3);

			fflush(out);
			terminal_read_input(i_buffer, sizeof(i_buffer));
			char *endptr = NULL;
			unsigned long selected = strtoul(i_buffer, &endptr, 10);
			if (endptr == i_buffer || *endptr != '\0' ||
			    selected > INT_MAX)
				selected = 0;

			if (simple) {
				const int p_chars =
					snprintf(NULL, 0, "%s (1-%zu): %s",
						 p_buf, c_count, i_buffer);
				const size_t p_len =
					(p_chars < 0) ? 0 : (size_t)p_chars;
				size_t p_padding = p_len < available ?
							   available - p_len :
							   0;

				while (p_padding--)
					ouputc(' ');
				ouwcol(M_SECTION);
				ouwbox(BOX_NORMAL, BOX_VERTICAL);
				ouwcol(COLOR_RESET);
			}

			ouputc('\n');

			if (selected >= 1 && selected <= c_count) {
				p.flags.content_printed = 1;
				return (int)selected - 1 +
				       PRINT_FIRST_CHOICE_INDEX__SUCCESS;
			}

			if (simple) {
				oprintf("%s %d and %zu.\n", w_msg, 1, c_count);
			} else {
				const int w_chars =
					snprintf(NULL, 0, "%s %s 1 and %zu.",
						 w_icon, w_msg, c_count);
				const size_t w_len =
					(w_chars < 0) ? 0 : (size_t)w_chars;
				const size_t w_padding =
					available > w_len ?
						available - w_len + 2 :
						0;

				ouwcol(M_SECTION);
				ouwbox(BOX_NORMAL, BOX_VERTICAL);
				oprintf("%s %s %s %d and %zu.%*s", w_color,
					w_icon, w_msg, 1, c_count,
					(int)w_padding, "");
				ouwcol(M_SECTION);
				ouwbox(BOX_NORMAL, BOX_VERTICAL);
				ouwcol(COLOR_RESET);
				ouputc('\n');
			}
		} while (1);
	} else if (type == M_PROMPT) {
		char *result = arg.uinput.out;
		const size_t rsz = arg.uinput.out_size;
		if (rsz < 2) {
#if DEFINE_AS_1_TO_TURN_OFF_DEV_MESSAGES == 0
			print_error_context("_TO_DEV_");
			print(M_NONE, ERROR "Input buffer size is too small.");
#endif
			return PRINT_PROMPT_BUFFER_SIZE_SHOULD_BE_2_OR_MORE__ERROR;
		}

		if (!simple) {
			ouwcol(M_SECTION);
			ouwbox(BOX_NORMAL, BOX_VERTICAL);
			ouwcol(type);
			ouputc(' ');
			ouwico(type);
			ouputc(' ');
		}

		ouwrite(p_buf, p_buflen);
		ouwrite(": ", 2);

		fflush(out);
		terminal_read_input(result, rsz);

		if (!simple) {
			const int p_chars =
				snprintf(NULL, 0, "%s: %s", p_buf, result);
			const size_t p_len = (p_chars < 0) ? 0 :
							     (size_t)p_chars;
			size_t p_padding =
				p_len < available ? available - p_len : 0;

			while (p_padding--)
				ouputc(' ');
			ouwcol(M_SECTION);
			ouwbox(BOX_NORMAL, BOX_VERTICAL);
			ouwcol(COLOR_RESET);
		}

		ouputc('\n');
		p.flags.content_printed = 1;

		goto cleanup;
	} else {
		if (simple) {
			ouputs(p_buf);
		} else {
			ouwcol(M_SECTION);
			ouwbox(BOX_NORMAL, BOX_VERTICAL);
			ouwcol(type);
			ouputc(' ');

			if (PICO(type) != ICON_NONE) {
				if (arg.loc != FIRST)
					ouwbox(BOX_NORMAL, arg.loc);
				else
					ouwico(type);

				ouputc(' ');
			}

			ouputs(p_buf);
			size_t padding =
				available > p_buflen ? available - p_buflen : 0;
			while (padding--)
				ouputc(' ');
			ouwcol(M_SECTION);
			ouwbox(BOX_NORMAL, BOX_VERTICAL);
			ouwcol(COLOR_RESET);
		}

		ouputc('\n');

		p.flags.content_printed = 1;
	}

cleanup:
	funlockfile(out);
	return PRINT_SUCCESS;
}
