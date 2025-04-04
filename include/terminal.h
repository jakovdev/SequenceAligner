#ifndef TERMINAL_H
#define TERMINAL_H

#include "arch.h"
#include <ctype.h>

INLINE int
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

INLINE void
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

INLINE void
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
    term.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &term);
#endif
}

INLINE void
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

INLINE void
terminal_read_input(char* input_buffer, int buffer_size, int* choice)
{
    int idx = 0;
    int c;

    fflush(stdout);

    terminal_mode_raw();

    while (1)
    {
        c = getchar();

        if (c == '\n' || c == '\r')
        {
            break;
        }

        if (isdigit(c) && idx < buffer_size - 1)
        {
            input_buffer[idx++] = c;
            input_buffer[idx] = '\0';
            printf("%c", c);
            fflush(stdout);
        }
    }

    terminal_mode_restore();
    *choice = atoi(input_buffer);
}

#endif // TERMINAL_H