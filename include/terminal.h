#ifndef TERMINAL_H
#define TERMINAL_H

#include "arch.h"

static inline int
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

static inline void
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

static inline void
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

static inline void
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

static inline void
terminal_read_input(char* input_buffer, int input_buffer_size)
{
    int input_character_index = 0;
    int input_character;

    fflush(stdout);

    terminal_mode_raw();

    while (true)
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

#endif // TERMINAL_H