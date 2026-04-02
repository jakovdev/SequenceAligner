#define PROGRESS_IMPLEMENTATION
#define PROGRESS_PRINT_H
#include "util/print.h"
#include "util/progress.h"

#include "util/args.h"

ARG_EXTERN(disable_write);
ARGUMENT(disable_progress) = {
	.set = &progress_disable,
	.help = "Disable progress bars",
	.lopt = "no-progress",
	.opt = 'P',
	.help_order = ARG_ORDER_AFTER(ARG(disable_write)),
};
