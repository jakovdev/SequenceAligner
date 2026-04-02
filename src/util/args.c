#define ARGS_IMPLEMENTATION
#define ARGS_PRINT_H
#include "util/print.h"
#include "util/args.h"

static struct arg_callback parse_help(const char *str, void *dest)
{
	(void)str;
	(void)dest;
	args_help_print("Usage: ", argr.v[0], " [ARGUMENTS]\n",
			"\nRequired arguments:\n", "\nOptional arguments:\n");
	exit(EXIT_SUCCESS);
}

ARGUMENT(help) = {
	.parse_callback = parse_help,
	.help = "Display this help message",
	.lopt = "help",
	.opt = 'h',
};
