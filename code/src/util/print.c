#include "util/print.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/args.h"

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
	HWND consoleWnd = GetConsoleWindow();
	DWORD consoleProcessId = 0;

	if (consoleWnd) {
		GetWindowThreadProcessId(consoleWnd, &consoleProcessId);

		if (consoleProcessId == GetCurrentProcessId()) {
			char path[MAX_PATH] = { 0 };
			char dir[MAX_PATH] = { 0 };
			char name[MAX_PATH] = { 0 };

			if (GetModuleFileNameA(NULL, path, MAX_PATH)) {
				char *slash = strrchr(path, '\\');
				if (slash) {
					size_t dirLen = (size_t)(slash - path);
					memcpy(dir, path, dirLen);
					dir[dirLen] = '\0';
					strcpy(name, slash + 1);
				} else {
					strcpy(name, path);
					dir[0] = '\0';
				}

				char cmd[2048];
				if (dir[0]) {
					snprintf(
						cmd, sizeof(cmd),
						"cmd.exe /k \"cd /d \"%s\" && \"%s\"\"",
						dir, name);
				} else {
					snprintf(cmd, sizeof(cmd),
						 "cmd.exe /k \"%s\"", name);
				}

				STARTUPINFOA si = { 0 };
				PROCESS_INFORMATION pi = { 0 };
				si.cb = sizeof(si);

				if (CreateProcessA(NULL, cmd, NULL, NULL, FALSE,
						   CREATE_NEW_CONSOLE, NULL,
						   dir[0] ? dir : NULL, &si,
						   &pi)) {
					CloseHandle(pi.hProcess);
					CloseHandle(pi.hThread);
					ExitProcess(0);
				}
			}
		}
	}

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

enum p_location {
	LOC_FIRST,
	LOC_MIDDLE,
	LOC_LAST,
};

enum p_type {
	T_NONE,
	T_INFO,
	T_VERBOSE,
	T_WARNING,
	T_ERROR,
	T_HEADER,
	T_SECTION,
	T_CHOICE,
	T_PROMPT,
	T_PROGRESS,
	T_TYPES
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
	size_t err_len;
	const struct {
		enum color color;
		enum icon icon;
		bool required;
	} map[T_TYPES];
	const char boxes[BOX_TYPE_COUNT][BOX_CHAR_COUNT][sizeof("╔")];
	char err_ctx[TERMINAL_WIDTH];
	const char progress_filled_char[sizeof("■")];
	const char progress_empty_char[sizeof("·")];
	const char ansi_escape_start[sizeof("\x1b")];
	char ansi_carriage_return[sizeof("\r")];
} p = {
    .map = {
        [T_NONE]     = { COLOR_RESET,       ICON_NONE,    false },
        [T_INFO]     = { COLOR_BLUE,        ICON_INFO,    false },
        [T_VERBOSE]  = { COLOR_GRAY,        ICON_DOT,     true  },
        [T_WARNING]  = { COLOR_YELLOW,      ICON_WARNING, true  },
        [T_ERROR]    = { COLOR_RED,         ICON_ERROR,   true  },
        [T_HEADER]   = { COLOR_BRIGHT_CYAN, ICON_NONE,    false },
        [T_SECTION]  = { COLOR_BLUE,        ICON_NONE,    false },
        [T_CHOICE]   = { COLOR_BLUE,        ICON_INFO,    true  },
        [T_PROMPT]   = { COLOR_BLUE,        ICON_INFO,    true  },
        [T_PROGRESS] = { COLOR_BRIGHT_CYAN, ICON_ARROW,   false },
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

static bool force;
static bool verbose;
static bool quiet;
static bool nodetail;
static bool in_section;
static bool content_printed;
static bool is_init;

void perr_context(const char *context)
{
	if (!context) {
		p.err_ctx[0] = '\0';
		p.err_len = 0;
		return;
	}

	int len = snprintf(p.err_ctx, sizeof(p.err_ctx), "%s | ", context);
	if (len < 0) {
		p.err_ctx[0] = '\0';
		p.err_len = 0;
		return;
	}

	p.err_len = (size_t)len;
	if (p.err_len >= sizeof(p.err_ctx)) {
		p.err_len = sizeof(p.err_ctx) - 1;
		p.err_ctx[p.err_len] = '\0';
	}
}

void print_streams(FILE *in, FILE *out, FILE *err)
{
	p.in = in;
	p.out = out;
	p.err = err;
}

static void print_section_end(void)
{
	if (in_section)
		psection_end();
}

static void print_init(void)
{
	terminal_init();
	if (!terminal_environment()) {
		p.ansi_carriage_return[0] = '\n';
		nodetail = true;
	}

	print_streams(stdin, stdout, stderr);
	atexit(print_section_end);
	is_init = true;
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
	if (force)
		return true;

	char result[2] = { 0 };
	pinput_s(result, "%s [y/N]", prompt);
	return result[0] == 'y' || result[0] == 'Y';
}

bool print_Yn(const char *P_RESTRICT prompt)
{
	if (force)
		return true;

	char result[2] = { 0 };
	pinput_s(result, "%s [Y/n]", prompt);
	return !(result[0] == 'n' || result[0] == 'N');
}

bool print_yn(const char *P_RESTRICT prompt)
{
	if (force)
		return true;

	char result[2] = { 0 };
repeat:
	pinput_s(result, "%s [y/n]", prompt);
	if (result[0] == 'y' || result[0] == 'Y')
		return true;
	else if (result[0] == 'n' || result[0] == 'N')
		return false;

	goto repeat;
}

#define ouputc(c) fputc_unlocked((c), out)
#define oprintf(...) fprintf(out, __VA_ARGS__)
#define ouwrite(buf, size) fwrite_unlocked((buf), 1, (size), out)
#define ouwico(t) ouwrite(p.icons[PICO(t)], PICOSIZ(t))
#define ouwcol(t) ouwrite(p.codes[PCOL(t)], PCOLSIZ(t))
#define ouwbox(bt, bpart) ouwrite(p.boxes[(bt)][bpart], BOXSIZ)

static int last_percentage = -1;

enum p_return print(const char *P_RESTRICT fmt, ...)
{
	if (!is_init)
		print_init();

	FILE *out = p.out;

	char p_buf[BUFSIZ] = { 0 };
	int p_bufsiz = 0;

	enum p_location loc = LOC_FIRST;
	enum p_type type = T_NONE;

	if (!fmt) { /* Section end only */
		type = T_SECTION;
		goto skip_fmt;
	}

	while (*fmt) {
		unsigned char c = (unsigned char)*fmt;
		if (c >= T_INFO && c <= T_SECTION) {
			type = (enum p_type)c;
			fmt++;
		} else if (c == 0x11) {
			loc = LOC_MIDDLE;
			fmt++;
		} else if (c == 0x12) {
			loc = LOC_LAST;
			fmt++;
		} else {
			break;
		}
	}

	if (type == T_ERROR)
		out = p.err;

	if ((quiet && !p.map[type].required) || (type == T_VERBOSE && !verbose))
		return PRINT_SKIPPED_BECAUSE_QUIET_OR_VERBOSE_NOT_ENABLED__SUCCESS;

	if (!in_section && type != T_HEADER && type != T_SECTION)
		psection();

	if (last_percentage != -1 && content_printed) {
		fputc('\n', out);
		last_percentage = -1;
	}

	{
		va_list v_args;
		va_start(v_args, fmt);
		p_bufsiz = vsnprintf(p_buf, sizeof(p_buf), fmt, v_args);
		va_end(v_args);

		if (p_bufsiz < 0) {
			pdev("Failed to format string");
			return PRINT_INVALID_FORMAT_ARGS__ERROR;
		}
	}

skip_fmt:
	bool simple = nodetail || (quiet && p.map[type].required);
	const size_t available = p.width - 3 - (!PICO(type) ? 0 : 2);
	size_t p_buflen = (size_t)p_bufsiz;
	if (p_buflen > available) { /* Overflow, no box/icon/color then */
		pdev("Message too long, doing a simple print");
		simple = true;
	}

	flockfile(out);

	if (type == T_HEADER) {
		if (!fmt)
			goto cleanup;

		if (simple) {
			ouputc('\n');
			ouwrite(p_buf, p_buflen);
			ouwrite("\n\n", 2);
			in_section = false;
			goto cleanup;
		}

		if (in_section) {
			funlockfile(out);
			psection_end();
			flockfile(out);
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
		in_section = false;
	} else if (type == T_SECTION) {
		if (in_section && (!fmt || content_printed)) {
			if (!simple) {
				ouwcol(T_SECTION);
				ouwbox(BOX_NORMAL, BOX_BOTTOM_LEFT);

				size_t iw;
				for (iw = 0; iw < p.width - 2; iw++)
					ouwbox(BOX_NORMAL, BOX_HORIZONTAL);

				ouwbox(BOX_NORMAL, BOX_BOTTOM_RIGHT);
				ouwcol(COLOR_RESET);
			}

			ouputc('\n');
			in_section = false;
			content_printed = false;
		}

		if (!fmt)
			goto cleanup;

		if (simple) {
			ouwrite(p_buf, p_buflen);
			ouputc('\n');

			in_section = true;
			content_printed = false;
			goto cleanup;
		}

		size_t l_dashes = 2;
		size_t r_dashes = p.width - 2 - l_dashes - p_buflen - 2;

		ouwcol(T_SECTION);
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

		in_section = true;
		content_printed = false;
	} else {
		if (simple) {
			if (type == T_ERROR && p.err_ctx[0])
				ouwrite(p.err_ctx, p.err_len);
			ouwrite(p_buf, p_buflen);
			ouputc('\n');
			content_printed = true;
			goto cleanup;
		}

		ouwcol(T_SECTION);
		ouwbox(BOX_NORMAL, BOX_VERTICAL);
		ouwcol(type);
		ouputc(' ');

		if (PICO(type) != ICON_NONE) {
			if (loc != LOC_FIRST)
				ouwbox(BOX_NORMAL, loc);
			else
				ouwico(type);

			ouputc(' ');
		}

		size_t padding = available - p_buflen;
		if (type == T_ERROR && p.err_ctx[0]) {
			ouwrite(p.err_ctx, p.err_len);
			padding -= p.err_len;
		}
		ouwrite(p_buf, p_buflen);
		while (padding--)
			ouputc(' ');
		ouwcol(T_SECTION);
		ouwbox(BOX_NORMAL, BOX_VERTICAL);
		ouwcol(COLOR_RESET);
		ouputc('\n');
		content_printed = true;
	}

cleanup:
	funlockfile(out);
	return PRINT_SUCCESS;
}

enum p_return progress_bar(int percent, const char *P_RESTRICT fmt, ...)
{
	if (!is_init)
		print_init();

	FILE *out = p.out;

	if (quiet)
		return PRINT_SKIPPED_BECAUSE_QUIET_OR_VERBOSE_NOT_ENABLED__SUCCESS;

	if (!in_section)
		psection();

#define CLAMP(val, min_val, max_val) (min(max((val), (min_val)), (max_val)))
	int p_percent = CLAMP(percent, 0, 100);
	if (p_percent == last_percentage ||
	    (p_percent == 100 && last_percentage == -1))
		return PRINT_REPEAT_PROGRESS_PERCENT__SUCCESS;

	last_percentage = p_percent;
	if (last_percentage == 100)
		last_percentage = -1;

	char p_buf[BUFSIZ] = { 0 };
	int p_bufsiz = 0;

	if (fmt) {
		va_list v_args;
		va_start(v_args, fmt);
		p_bufsiz = vsnprintf(p_buf, sizeof(p_buf), fmt, v_args);
		va_end(v_args);

		if (p_bufsiz < 0) {
			pdev("Failed to format string");
			return PRINT_INVALID_FORMAT_ARGS__ERROR;
		}
	}

	bool simple = nodetail;
	const size_t available = p.width - 3 - (!PICO(T_PROGRESS) ? 0 : 2);
	size_t p_buflen = (size_t)p_bufsiz;
	if (p_buflen > available) {
		pdev("Progress bar message too long, doing a simple print");
		simple = true;
	}

	flockfile(out);

	const int digits = p_percent < 10 ? 1 : (p_percent < 100 ? 2 : 3);

	if (simple) {
		if (in_section)
			ouwrite(p.ansi_carriage_return, 1);

		ouwrite(p_buf, p_buflen);
		ouputc(' ');
		if (p_percent == 100) {
			ouwrite("100%\n", 5);
		} else {
			if (p_percent >= 10)
				ouputc('0' + p_percent / 10);
			ouputc('0' + p_percent % 10);
			ouputc('%');
		}

		fflush(out);
		content_printed = true;
		goto cleanup;
	}

	const size_t meta_width = (size_t)digits + 2 + 1 + 1 + 1;
	const size_t bar_width = available - p_buflen - meta_width - 1;
	const size_t filled_width = bar_width * (size_t)p_percent / 100;
	const size_t empty_width = bar_width - filled_width;

	if (in_section)
		ouwrite(p.ansi_carriage_return, 1);

	ouwcol(T_SECTION);
	ouwbox(BOX_NORMAL, BOX_VERTICAL);
	if (p_percent % 2)
		ouwcol(COLOR_CYAN);
	else
		ouwcol(T_PROGRESS);

	ouputc(' ');
	ouwico(T_PROGRESS);
	ouputc(' ');
	ouwrite(p_buf, p_buflen);
	ouwrite(" [", 2);

	size_t ifew;
	for (ifew = 0; ifew < filled_width; ifew++)
		ouwrite(p.progress_filled_char, 3);

	for (ifew = 0; ifew < empty_width; ifew++)
		ouwrite(p.progress_empty_char, 2);

	ouwrite("] ", 2);
	if (p_percent == 100) {
		ouwrite("100% ", 5);
	} else {
		if (p_percent >= 10)
			ouputc('0' + p_percent / 10);
		ouputc('0' + p_percent % 10);
		ouwrite("% ", 2);
	}

	ouwcol(T_SECTION);
	ouwbox(BOX_NORMAL, BOX_VERTICAL);
	ouwcol(COLOR_RESET);
	if (p_percent == 100)
		ouputc('\n');

	fflush(out);
	content_printed = true;

cleanup:
	funlockfile(out);
	return PRINT_SUCCESS;
}

enum p_return input(P_INPUT in, size_t size, const char *P_RESTRICT fmt, ...)
{
	if (!is_init)
		print_init();

	if (!fmt) {
		pdev("Input format string is NULL");
		return PRINT_INVALID_FORMAT_ARGS__ERROR;
	}

	enum p_type type = T_NONE;
	while (*fmt) {
		unsigned char c = (unsigned char)*fmt;
		if (c >= 0x07 && c <= 0x08) {
			type = (enum p_type)c;
			fmt++;
		} else {
			break;
		}
	}
	if (type != T_CHOICE && type != T_PROMPT) {
		pdev("Invalid input type");
		return PRINT_INVALID_INPUT_TYPE__ERROR;
	}

	FILE *out = p.out;

	if (!in_section)
		psection();

	if (last_percentage != -1 && content_printed) {
		fputc('\n', out);
		last_percentage = -1;
	}

	char p_buf[BUFSIZ] = { 0 };
	int p_bufsiz = 0;

	{
		va_list v_args;
		va_start(v_args, fmt);
		p_bufsiz = vsnprintf(p_buf, sizeof(p_buf), fmt, v_args);
		va_end(v_args);

		if (p_bufsiz < 0) {
			pdev("Failed to format string");
			return PRINT_INVALID_FORMAT_ARGS__ERROR;
		}
	}

	bool simple = nodetail;
	const size_t available = p.width - 3 - (!PICO(type) ? 0 : 2);
	size_t p_buflen = (size_t)p_bufsiz;
	if (p_buflen > available) {
		pdev("Input prompt too long, doing a simple print");
		simple = true;
	}

	flockfile(out);

	if (type == T_CHOICE) {
		char **choices = in.choices;
		size_t c_count = size;

		if (c_count < 2) {
			funlockfile(out);
			pdev("Not enough choices (<2)");
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

				ouwcol(T_SECTION);
				ouwbox(BOX_NORMAL, BOX_VERTICAL);
				ouwcol(type);
				oprintf(" %zu: %s%*s", c + 1, choices[c],
					(int)padding, "");
				ouwcol(T_SECTION);
				ouwbox(BOX_NORMAL, BOX_VERTICAL);
				ouwcol(COLOR_RESET);
				ouputc('\n');
			}
		}

		char i_buffer[TERMINAL_WIDTH] = { 0 };

		do {
			if (!simple) {
				ouwcol(T_SECTION);
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
			errno = 0;
			char *endptr = NULL;
			unsigned long selected = strtoul(i_buffer, &endptr, 10);
			if (endptr == i_buffer || *endptr != '\0' ||
			    errno == ERANGE || selected > INT_MAX)
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
				ouwcol(T_SECTION);
				ouwbox(BOX_NORMAL, BOX_VERTICAL);
				ouwcol(COLOR_RESET);
			}

			ouputc('\n');

			if (selected >= 1 && selected <= c_count) {
				content_printed = true;
				funlockfile(out);
				return (int)selected - 1 +
				       PRINT_FIRST_CHOICE_INDEX__SUCCESS;
			}

			funlockfile(out);
			pwarn("Please enter a number between 1 and %zu",
			      c_count);
			flockfile(out);
		} while (1);
	} else if (type == T_PROMPT) {
		char *result = in.output;
		const size_t rsz = size;
		if (rsz < 2) {
			funlockfile(out);
			pdev("Input buffer size is too small");
			return PRINT_PROMPT_BUFFER_SIZE_SHOULD_BE_2_OR_MORE__ERROR;
		}

		if (!simple) {
			ouwcol(T_SECTION);
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
			ouwcol(T_SECTION);
			ouwbox(BOX_NORMAL, BOX_VERTICAL);
			ouwcol(COLOR_RESET);
		}

		ouputc('\n');
		content_printed = true;
	}

	funlockfile(out);
	return PRINT_SUCCESS;
}

enum p_return pdev(const char *fmt, ...)
{
#ifndef NDEBUG
	va_list v_args;
	va_start(v_args, fmt);
	char buf[BUFSIZ];
	vsnprintf(buf, sizeof(buf), fmt, v_args);
	va_end(v_args);

	perr_context("_TO_DEV_");
	return perr("%s", buf);
#else
	(void)fmt;
	return PRINT_TO_DEV_NDEBUG__ERROR;
#endif
}

ARGUMENT(force) = {
	.opt = 'F',
	.lopt = "force-proceed",
	.help = "Force proceed without user prompts (for CI)",
	.set = &force,
};

ARGUMENT(verbose) = {
	.opt = 'v',
	.lopt = "verbose",
	.help = "Enable verbose printing",
	.set = &verbose,
};

ARGUMENT(quiet) = {
	.opt = 'q',
	.lopt = "quiet",
	.help = "Suppress all non-error printing",
	.set = &quiet,
};

ARGUMENT(disable_detail) = {
	.opt = 'D',
	.lopt = "no-detail",
	.help = "Disable detailed printing",
	.set = &nodetail,
};
