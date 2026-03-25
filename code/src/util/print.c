#define PRINT_IMPLEMENTATION
#include "util/print.h"

#include "util/args.h"

ARG_EXTERN(disable_progress);
ARGUMENT(print_nodetail) = {
	.set = &print_nodetail,
	.help = "Disable detailed printing",
	.lopt = "no-detail",
	.opt = 'D',
	.help_order = ARG_ORDER_AFTER(ARG(disable_progress)),
};

ARGUMENT(print_force) = {
	.set = &print_force,
	.help = "Force proceed without user prompts (for CI)",
	.lopt = "force-proceed",
	.opt = 'F',
	.help_order = ARG_ORDER_AFTER(ARG(print_nodetail)),
};

ARGUMENT(print_quiet) = {
	.set = &print_quiet,
	.help = "Suppress all non-error printing",
	.lopt = "quiet",
	.opt = 'Q',
	.help_order = ARG_ORDER_AFTER(ARG(print_force)),
};

ARGUMENT(print_verbose) = {
	.set = &print_verbose,
	.help = "Enable verbose printing",
	.lopt = "verbose",
	.opt = 'V',
	.help_order = ARG_ORDER_AFTER(ARG(print_quiet)),
};
