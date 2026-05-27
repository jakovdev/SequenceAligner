#include "bio/alignment.h"

#include <args.h>
#include <print.h>
#include <progress.h>
#include <string.h>

#include "bio/sequence.h"
#include "io/input.h"
#include "io/output.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

struct align_method ALIGN_METHODS[ALIGN_COUNT];
enum align_methods METHOD_ID = ALIGN_INVALID;

s32 GAP_PEN;
s32 GAP_OPN;
s32 GAP_EXT;

size_t TABLE_SIZE;

bool align(const struct input *dataset, struct output *sm)
{
	size_t total = (size_t)dataset->alignments;
	pinfo("Performing %zu pairwise alignments", total);
	if (!progress_start(total, THREAD_NUM, "Aligning sequences"))
		return false;

	s32 seqs_n = dataset->seqs_n;
	TABLE_SIZE = (dataset->lengths_max + 1) * (dataset->lengths_max + 1);
	const struct sequence *restrict seqs = dataset->seqs;
	auto method = ALIGN_METHODS[METHOD_ID].method;
	bench_align_start();
#pragma omp parallel
	{
		s32 *MALLOCA_AL(table, CACHE_LINE, 3 * TABLE_SIZE);
		s32 *MALLOCA_AL(ind, CACHE_LINE, dataset->lengths_max);
		s32 *MALLOCA_AL(cols, CACHE_LINE, (size_t)seqs_n);
		if unlikely (!table || !ind || !cols) {
			perr("Out of memory allocating alignment buffers");
			exit(EXIT_FAILURE);
		}
#pragma omp for schedule(dynamic)
		for (s32 col = 1; col < seqs_n; col++) {
			const struct sequence *restrict seq = &seqs[col];
			for (s32 i = 0; i < seq->length; ++i)
				ind[i] = SEQ_LUT[(uchar)seq->letters[i]];
			for (s32 row = 0; row < col; row++)
				cols[row] = method(seq, &seqs[row], ind, table);

			output_fill(sm, cols, col);
			progress_add((size_t)col);
		}

		progress_flush();
		free_aligned(cols);
		free_aligned(ind);
		free_aligned(table);
	}

	bench_align_end();
	progress_end();
	bench_align_print();
	return true;
}

static char help[512];

[[gnu::constructor]]
static void build_help_strings(void)
{
	snprintf(help, sizeof(help), "Alignment method\n");
	for (int i = 0; i < ALIGN_COUNT; i++) {
		const char *n = (i == ALIGN_COUNT - 1) ? "" : "\n";
		size_t len = strlen(help);
		snprintf(help + len, sizeof(help) - len, "  %s: %s%s",
			 ALIGN_METHODS[i].aliases[0], ALIGN_METHODS[i].name, n);
	}
}

static struct arg_callback parse_align_method(const char *str, void *)
{
	METHOD_ID = ALIGN_INVALID;
	errno = 0;
	char *endptr = {};
	long id = strtol(str, &endptr, 10);
	if (endptr != str && *endptr == '\0' && errno != ERANGE &&
	    id > ALIGN_INVALID && id < ALIGN_COUNT)
		METHOD_ID = (enum align_methods)id;

	for (int i = 0; METHOD_ID == ALIGN_INVALID && i < ALIGN_COUNT; i++) {
		for (const char **a = ALIGN_METHODS[i].aliases; *a; a++) {
			if (strcasecmp(str, *a) == 0) {
				METHOD_ID = (enum align_methods)i;
				break;
			}
		}
	}

	if (METHOD_ID == ALIGN_INVALID)
		return ARG_INVALID("Invalid alignment method");

	return ARG_VALID();
}

static void print_config_method(void)
{
	pinfom("Method: %s", ALIGN_METHODS[METHOD_ID].name);
}

ARG_EXTERN(substitution_matrix);

ARGUMENT(alignment_method) = {
	.opt = 'a',
	.lopt = "align",
	.help = help,
	.param = "METHOD",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.parse_callback = parse_align_method,
	.action_callback = print_config_method,
	.action_order = ARG_ORDER_AFTER(ARG(substitution_matrix)),
	.help_order = ARG_ORDER_AFTER(ARG(substitution_matrix)),
};

ARG_PARSE_L(gap_value, 10, s32, -(s32), (val < 0 || val > INT_MAX),
	    "Gap values must be positive integers")

static struct arg_callback validate_gap_pen(void)
{
	if (ALIGN_METHODS[METHOD_ID].gap == GAP_LINEAR)
		return ARG_VALID();
	return ARG_INVALID("Gap penalty cannot be set for non-linear methods");
}

static struct arg_callback validate_gap_affine(void)
{
	if (METHOD_ID == ALIGN_GA && GAP_OPN == GAP_EXT &&
	    print_Yn("Equal affine gaps found, switch to Needleman-Wunsch?")) {
		METHOD_ID = ALIGN_NW;
		GAP_PEN = GAP_OPN;
		GAP_OPN = SCORE_MIN;
		GAP_EXT = SCORE_MIN;
		return ARG_VALID();
	}

	if (ALIGN_METHODS[METHOD_ID].gap == GAP_AFFINE)
		return ARG_VALID();

	return ARG_INVALID("Affine gaps cannot be set for non-affine methods");
}

static void print_config_gaps(void)
{
	switch (ALIGN_METHODS[METHOD_ID].gap) {
	case GAP_LINEAR:
		pinfom("Gap penalty: %d", GAP_PEN);
		break;
	case GAP_AFFINE:
		pinfom("Gap open: %d, extend: %d", GAP_OPN, GAP_EXT);
		break;
	default:
		unreachable_release();
	}
}

ARG_DECLARE(gap_open);
ARG_DECLARE(gap_extend);

ARGUMENT(gap_penalty) = {
	.opt = 'p',
	.lopt = "gap-penalty",
	.help = "Linear gap penalty",
	.param = "N",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = &GAP_PEN,
	.parse_callback = parse_gap_value,
	.validate_callback = validate_gap_pen,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_AFTER(ARG(gap_open)),
	.action_callback = print_config_gaps,
	.action_order = ARG_ORDER_AFTER(ARG(alignment_method)),
	.help_order = ARG_ORDER_AFTER(ARG(alignment_method)),
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(alignment_method)),
	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(gap_open), ARG(gap_extend)),
};

ARGUMENT(gap_open) = {
	.opt = 's',
	.lopt = "gap-open",
	.help = "Affine gap open penalty",
	.param = "N",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = &GAP_OPN,
	.parse_callback = parse_gap_value,
	.validate_callback = validate_gap_affine,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_AFTER(ARG(alignment_method)),
	.help_order = ARG_ORDER_AFTER(ARG(gap_penalty)),
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(alignment_method)),
	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(gap_penalty)),
};

ARGUMENT(gap_extend) = {
	.opt = 'e',
	.lopt = "gap-extend",
	.help = "Affine gap extend penalty",
	.param = "N",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = &GAP_EXT,
	.parse_callback = parse_gap_value,
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(alignment_method)),
	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(gap_penalty)),
};
