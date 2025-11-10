#include "util/print.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

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
#define fputc_unlocked _fputc_nolock
#define fwrite_unlocked _fwrite_nolock
#define flockfile _lock_file
#define funlockfile _unlock_file
#else
#include <termios.h>
#include <unistd.h>
#include <sys/param.h>
#define max MAX
#define min MIN
#endif

#define CLAMP(val, min_val, max_val) (min(max((val), (min_val)), (max_val)))

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

enum m_type {
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

enum color {
	COLOR_RESET,
	COLOR_UNDER,
	COLOR_RED,
	COLOR_GREEN,
	COLOR_YELLOW,
	COLOR_BLUE,
	COLOR_MAGENTA,
	COLOR_CYAN,
	COLOR_GRAY,
	COLOR_BRIGHT_CYAN,
	COLOR_TYPE_COUNT
};

enum icon {
	ICON_NONE,
	ICON_WARNING,
	ICON_DOT,
	ICON_INFO,
	ICON_ERROR,
	ICON_ARROW,
	ICON_TYPE_COUNT
};

enum requirement { P_OPTIONAL, P_REQUIRED };

enum box_pos {
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

enum box_type { BOX_NORMAL, BOX_FANCY, BOX_TYPE_COUNT };

static struct {
	const char *codes[COLOR_TYPE_COUNT];
	const char *icons[ICON_TYPE_COUNT];
	FILE *in;
	FILE *out;
	FILE *err;
	size_t width;
	const struct {
		enum color color;
		enum icon icon;
		enum requirement requirement;
	} map[M_TYPES];
	unsigned verbose : 1;
	unsigned quiet : 1;
	unsigned nodetail : 1;
	unsigned in_section : 1;
	unsigned content_printed : 1;
	unsigned is_init : 1;
	const char boxes[BOX_TYPE_COUNT][BOX_CHAR_COUNT][sizeof("╔")];
	char err_ctx[TERMINAL_WIDTH];
	const char progress_filled_char[sizeof("■")];
	const char progress_empty_char[sizeof("·")];
	const char ansi_escape_start[sizeof("\x1b")];
	char ansi_carriage_return[sizeof("\r")];
} p = {
    .map = {
        [0]          = { COLOR_RESET,       ICON_NONE,    P_OPTIONAL },
        [M_INFO]     = { COLOR_BLUE,        ICON_INFO,    P_OPTIONAL },
        [M_VERBOSE]  = { COLOR_GRAY,        ICON_DOT,     P_REQUIRED },
        [M_WARNING]  = { COLOR_YELLOW,      ICON_WARNING, P_REQUIRED },
        [M_ERROR]    = { COLOR_RED,         ICON_ERROR,   P_REQUIRED },
        [M_CHOICE]   = { COLOR_BLUE,        ICON_INFO,    P_REQUIRED },
        [M_PROMPT]   = { COLOR_BLUE,        ICON_INFO,    P_REQUIRED },
        [M_PROGRESS] = { COLOR_BRIGHT_CYAN, ICON_ARROW,   P_OPTIONAL },
        [M_HEADER]   = { COLOR_BRIGHT_CYAN, ICON_NONE,    P_OPTIONAL },
        [M_SECTION]  = { COLOR_BLUE,        ICON_NONE,    P_OPTIONAL },
    },
#define PCOL(t) p.map[(t)].color
#define PCOLSIZ(t) (PCOL(t) <= COLOR_UNDER ? 4 : 5)
    .codes = {
        [COLOR_RESET]       = "\x1b[0m",
	[COLOR_UNDER]       = "\x1b[4m",
        [COLOR_RED]         = "\x1b[31m",
        [COLOR_GREEN]       = "\x1b[32m",
        [COLOR_YELLOW]      = "\x1b[33m",
        [COLOR_BLUE]        = "\x1b[34m",
        [COLOR_MAGENTA]     = "\x1b[35m",
        [COLOR_CYAN]        = "\x1b[36m",
        [COLOR_GRAY]        = "\x1b[90m",
        [COLOR_BRIGHT_CYAN] = "\x1b[96m",
    },
#define PICO(t) p.map[(t)].icon
#define PICOSIZ(t) (PICO(t) >= ICON_ERROR ? 3 : PICO(t))
    .icons = {
        [ICON_NONE]    = "",
        [ICON_WARNING] = "!",
        [ICON_DOT]     = "·",
        [ICON_INFO]    = "•",
        [ICON_ERROR]   = "✗",
        [ICON_ARROW]   = "▶",
    },
#define BOXSIZ (sizeof(p.boxes[BOX_NORMAL][0]) - 1)
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
	p.verbose = !p.verbose;
}

void print_quiet_flip(void)
{
	p.quiet = !p.quiet;
}

void print_detail_flip(void)
{
	p.nodetail = !p.nodetail;
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

static void print_section_end(void)
{
	if (p.in_section)
		print(M_NONE, NULL);
}

static void print_init(void)
{
	terminal_init();
	if (!terminal_environment())
		p.ansi_carriage_return[0] = '\n';

	print_streams(stdin, stdout, stderr);
	atexit(print_section_end);
	p.is_init = 1;
}

static void terminal_read_input(char *buf, size_t buf_sz)
{
	if (!buf || !buf_sz)
		return;

	fflush(p.out);
	terminal_mode_raw();

	size_t i = 0;
	for (int c; (c = fgetc(p.in)) != EOF && c != '\n' && c != '\r';) {
		if (c == '\b' || c == 0x7F) {
			if (i) {
				while (i && ((buf[i - 1] & 0xC0) == 0x80))
					i--;
				buf[--i] = '\0';
				fwrite("\b \b", 1, 3, p.out);
				fflush(p.out);
			}
			continue;
		}

		if (i < buf_sz - 1 && (c >= 0x20 && c <= 0x7E)) {
			buf[i++] = (char)c;
			buf[i] = '\0';
			fputc(c, p.out);
			fflush(p.out);
		}
	}

	buf[i] = '\0';
	terminal_mode_restore();
}

bool print_yN(const char *P_RESTRICT prompt)
{
	char result[2] = { 0 };
	print(M_UINS(result) "%s [y/N]", prompt);
	return result[0] == 'y' || result[0] == 'Y';
}

bool print_Yn(const char *P_RESTRICT prompt)
{
	char result[2] = { 0 };
	print(M_UINS(result) "%s [Y/n]", prompt);
	return !(result[0] == 'n' || result[0] == 'N');
}

bool print_yn(const char *P_RESTRICT prompt)
{
repeat:
	char result[2] = { 0 };
	print(M_UINS(result) "%s [y/n]", prompt);
	if (result[0] == 'y' || result[0] == 'Y')
		return true;
	else if (result[0] == 'n' || result[0] == 'N')
		return false;

	goto repeat;
}

enum p_return print(M_ARG arg, const char *P_RESTRICT fmt, ...)
{
	if (!p.is_init)
		print_init();

	enum m_type type = 0;
	if (fmt && fmt[0] >= '0' && fmt[0] <= '9') {
		type = (enum m_type)fmt[0] - '0';
		fmt++;
	}

	FILE *out = (type == M_ERROR) ? p.err : p.out;
#define ouputc(c) fputc_unlocked((c), out)
#define oprintf(...) fprintf(out, __VA_ARGS__)
#define ouwrite(buf, size) fwrite_unlocked((buf), 1, (size), out)

	const bool is_verbose = type == M_VERBOSE && !p.verbose;
	const bool is_required = p.map[type].requirement == P_REQUIRED;
	if ((p.quiet && !is_required) || is_verbose)
		return PRINT_SKIPPED_BECAUSE_QUIET_OR_VERBOSE_NOT_ENABLED__SUCCESS;

	if (!p.in_section && type != M_HEADER && type != M_SECTION)
		print(M_NONE, SECTION);

	static int last_percentage = -1;
	if (type == M_PROGRESS) {
		const int percent = CLAMP(arg.percent, 0, 100);
		if (percent == last_percentage ||
		    (percent == 100 && last_percentage == -1))
			return PRINT_REPEAT_PROGRESS_PERCENT__SUCCESS;

		last_percentage = percent;

		if (last_percentage == 100)
			last_percentage = -1;
	} else if (last_percentage != -1 && p.content_printed) {
		fputc('\n', out);
		last_percentage = -1;
	}

#define ouwico(t) ouwrite(p.icons[PICO(t)], PICOSIZ(t))
#define ouwcol(t) ouwrite(p.codes[PCOL(t)], PCOLSIZ(t))
#define ouwbox(bt, bpart) ouwrite(p.boxes[(bt)][bpart], BOXSIZ)

	char p_buf[BUFSIZ] = { 0 };
	int p_bufsiz = 0;

	if (!fmt) { /* Section end only */
		type = M_SECTION;
		goto skip_fmt;
	}

	char fmt_copy[BUFSIZ];
	size_t fmt_i = 0;

	fmt_copy[0] = '\0';

	if (type == M_ERROR && p.err_ctx[0] != '\0') {
		size_t ctxlen = strnlen(p.err_ctx, sizeof(fmt_copy) - 1);
		if (ctxlen) {
			memcpy(fmt_copy + fmt_i, p.err_ctx, ctxlen);
			fmt_i += ctxlen;
		}
	}

	const char *fmt_src = fmt ? fmt : "";
	size_t copy_len = strnlen(fmt_src, sizeof(fmt_copy) - 1 - fmt_i);
	if (copy_len)
		memcpy(fmt_copy + fmt_i, fmt_src, copy_len);
	fmt_copy[fmt_i + copy_len] = '\0';

	va_list v_args;
	va_start(v_args, fmt);
	p_bufsiz = vsnprintf(p_buf, sizeof(p_buf), fmt_copy, v_args);
	va_end(v_args);

	if (p_bufsiz < 0) {
#ifndef NDEBUG
		print_error_context("_TO_DEV_");
		print(M_NONE, ERR "Failed to format string");
#endif
		return PRINT_INVALID_FORMAT_ARGS__ERROR;
	}

skip_fmt:
	bool simple = p.nodetail || (p.quiet && is_required);
	const size_t available = p.width - 3 - (!PICO(type) ? 0 : 2);
	size_t p_buflen = (size_t)p_bufsiz;
	if (p_buflen > available) { /* Overflow, no box/icon/color then */
#ifndef NDEBUG
		print_error_context("_TO_DEV_");
		print(M_NONE, ERR "Message too long, doing a simple print");
#endif
		simple = true;
	}

	flockfile(out);

	if (type == M_HEADER) {
		if (!fmt)
			goto cleanup;

		if (simple) {
			ouputc('\n');
			ouwrite(p_buf, p_buflen);
			ouwrite("\n\n", 2);
			p.in_section = 0;
			goto cleanup;
		}

		if (p.in_section) {
			funlockfile(out);
			print(M_NONE, NULL);
		}

		ouwcol(type);
		ouwbox(BOX_FANCY, BOX_TOP_LEFT);

		size_t iwlrp;
		for (iwlrp = 0; iwlrp < p.width - 2; iwlrp++)
			ouwbox(BOX_FANCY, BOX_HORIZONTAL);

		ouwbox(BOX_FANCY, BOX_TOP_RIGHT);
		ouwcol(COLOR_RESET);
		ouputc('\n');

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

		ouwcol(type);
		ouwbox(BOX_FANCY, BOX_BOTTOM_LEFT);
		for (iwlrp = 0; iwlrp < p.width - 2; iwlrp++)
			ouwbox(BOX_FANCY, BOX_HORIZONTAL);

		ouwbox(BOX_FANCY, BOX_BOTTOM_RIGHT);
		ouwcol(COLOR_RESET);
		ouputc('\n');

		p.in_section = 0;

		goto cleanup;
	} else if (type == M_SECTION) {
		if (p.in_section && (!fmt || p.content_printed)) {
			if (!simple) {
				ouwcol(M_SECTION);
				ouwbox(BOX_NORMAL, BOX_BOTTOM_LEFT);

				size_t iw;
				for (iw = 0; iw < p.width - 2; iw++)
					ouwbox(BOX_NORMAL, BOX_HORIZONTAL);

				ouwbox(BOX_NORMAL, BOX_BOTTOM_RIGHT);
				ouwcol(COLOR_RESET);
			}

			ouputc('\n');
			p.in_section = 0;
			p.content_printed = 0;
		}

		if (!fmt)
			goto cleanup;

		if (simple) {
			ouwrite(p_buf, p_buflen);
			ouputc('\n');

			p.in_section = 1;
			p.content_printed = 0;
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
			ouwrite(p.codes[COLOR_UNDER], PCOLSIZ(COLOR_UNDER));
			ouwrite(p_buf, p_buflen);
			ouwcol(COLOR_RESET);
			ouwcol(type);
			ouputc(' ');
		}

		while (r_dashes--)
			ouwbox(BOX_NORMAL, BOX_HORIZONTAL);

		ouwbox(BOX_NORMAL, BOX_TOP_RIGHT);
		ouwcol(COLOR_RESET);
		ouputc('\n');

		p.in_section = 1;
		p.content_printed = 0;
		goto cleanup;
	} else if (type == M_PROGRESS) {
		const int percent = CLAMP(arg.percent, 0, 100);
		const int percent_width =
			percent < 10 ? 1 : (percent < 100 ? 2 : 3);

		if (simple) {
			if (p.in_section)
				ouwrite(p.ansi_carriage_return, 1);

			ouwrite(p_buf, p_buflen);
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
			p.content_printed = 1;
			goto cleanup;
		}

		const size_t meta_width = (size_t)percent_width + 2 + 1 + 1 + 1;
		const size_t bar_width = available - p_buflen - meta_width - 1;
		const size_t filled_width = bar_width * (size_t)percent / 100;
		const size_t empty_width = bar_width - filled_width;

		if (p.in_section)
			ouwrite(p.ansi_carriage_return, 1);

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
			ouwrite(p.progress_filled_char, 3);

		for (ifew = 0; ifew < empty_width; ifew++)
			ouwrite(p.progress_empty_char, 2);

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
		p.content_printed = 1;

		goto cleanup;
	} else if (type == M_CHOICE) {
		char **choices = arg.choice.choices;
		size_t c_count = arg.choice.n;

		if (c_count < 2) {
			funlockfile(out);
#ifndef NDEBUG
			print_error_context("_TO_DEV_");
			print(M_NONE, ERR "Not enough choices (<2)");
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

			funlockfile(out);
			terminal_read_input(i_buffer, sizeof(i_buffer));
			flockfile(out);
			char *endptr = NULL;
			unsigned long selected = strtoul(i_buffer, &endptr, 10);
			if (endptr == i_buffer || *endptr != '\0' ||
			    selected > INT_MAX)
				selected = 0;

			if (!simple) {
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
				p.content_printed = 1;
				funlockfile(out);
				return (int)selected - 1 +
				       PRINT_FIRST_CHOICE_INDEX__SUCCESS;
			}

			funlockfile(out);
			print(M_NONE,
			      WARNING
			      "Please enter a number between 1 and %zu.",
			      c_count);
			flockfile(out);
		} while (1);
	} else if (type == M_PROMPT) {
		char *result = arg.uinput.out;
		const size_t rsz = arg.uinput.out_size;
		if (rsz < 2) {
			funlockfile(out);
#ifndef NDEBUG
			print_error_context("_TO_DEV_");
			print(M_NONE, ERR "Input buffer size is too small.");
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

		funlockfile(out);
		terminal_read_input(result, rsz);
		flockfile(out);

		if (!simple) {
			const size_t p_len = p_buflen + 2 + strlen(result);
			size_t p_padding =
				p_len < available ? available - p_len : 0;

			while (p_padding--)
				ouputc(' ');
			ouwcol(M_SECTION);
			ouwbox(BOX_NORMAL, BOX_VERTICAL);
			ouwcol(COLOR_RESET);
		}

		ouputc('\n');
		p.content_printed = 1;

		goto cleanup;
	} else {
		if (simple) {
			ouwrite(p_buf, p_buflen);
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

			ouwrite(p_buf, p_buflen);
			size_t padding =
				available > p_buflen ? available - p_buflen : 0;
			while (padding--)
				ouputc(' ');
			ouwcol(M_SECTION);
			ouwbox(BOX_NORMAL, BOX_VERTICAL);
			ouwcol(COLOR_RESET);
		}

		ouputc('\n');

		p.content_printed = 1;
	}

#undef ouwico
#undef ouwcol
#undef ouwbox
cleanup:
	funlockfile(out);
	return PRINT_SUCCESS;
}
