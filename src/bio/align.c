#include "bio/alignment.h"

#include <args.h>
#include <print.h>
#include <progress.h>
#include <string.h>

#include "io/input.h"
#include "io/output.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

s32 GAP_PEN;
s32 GAP_OPN;
s32 GAP_EXT;

size_t TABLE_SIZE;
const struct align *ALIGN;

bool align(const struct input *in, const struct output *out)
{
	s32 num = in->num;
	size_t alignments = alignments((size_t)num);
	pinfo("Performing %zu pairwise alignments", alignments);
	if (!progress_start(alignments, THREAD_NUM, "Aligning sequences"))
		return false;

	TABLE_SIZE = (in->max + 1) * (in->max + 1);
	auto method = ALIGN->method;
	bench_align_start();
#pragma omp parallel
	{
		s32 *MALLOCA_AL(table, CACHE_LINE, 3 * TABLE_SIZE);
		s32 *MALLOCA_AL(ind, CACHE_LINE, in->max);
		s32 *MALLOCA_AL(cols, CACHE_LINE, (size_t)num);
		if (!table || !ind || !cols) {
			perr("Out of memory allocating alignment buffers");
			exit(EXIT_FAILURE);
		}
#pragma omp for schedule(dynamic)
		for (s32 col = 1; col < num; col++) {
			struct meta m1 = in->meta[col];
			s32 l1 = m1.len;
			seq s1 = in->letters + m1.off;
			for (s32 i = 0; i < l1; ++i)
				ind[i] = SEQ_LUT[s1[i]];
			for (s32 row = 0; row < col; row++) {
				struct meta m2 = in->meta[row];
				s32 l2 = m2.len;
				seq s2 = in->letters + m2.off;
				cols[row] = method(l1, l2, s2, ind, table);
			}

			output_fill(out, cols, col);
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
	for (auto m = __start_aligns; m < __stop_aligns; m++) {
		size_t len = strlen(help);
		snprintf(help + len, sizeof(help) - len, "  %s: %s\n",
			 *m->aliases, m->aliases[1]);
	}
}

static struct arg_callback parse_align(const char *str, void *)
{
	for (ALIGN = __start_aligns; ALIGN < __stop_aligns; ALIGN++) {
		for (const char **a = ALIGN->aliases; *a; a++) {
			if (strcasecmp(str, *a) == 0)
				return ARG_VALID();
		}
	}
	return ARG_INVALID("Invalid alignment method");
}

static struct arg_callback validate_align(void)
{
	return ALIGN->validate ? ALIGN->validate() : ARG_VALID();
}

static void print_align(void)
{
	pinfom("Method: %s", *ALIGN->aliases);
}

ARG_EXTERN(substitution_matrix);
ARG_EXTERN(gap_penalty);

ARGUMENT(align) = {
	.opt = 'a',
	.lopt = "align",
	.help = help,
	.param = "METHOD",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.parse_callback = parse_align,
	.validate_callback = validate_align,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_AFTER(ARG(gap_penalty)),
	.action_callback = print_align,
	.action_order = ARG_ORDER_AFTER(ARG(substitution_matrix)),
	.help_order = ARG_ORDER_AFTER(ARG(substitution_matrix)),
};

ARG_PARSE_L(gap_value, 10, s32, -(s32), (val < 0 || val > S32_MAX),
	    "Gap values must be positive integers")

static struct arg_callback validate_gap_pen(void)
{
	if (ALIGN->gap == GAP_LINEAR)
		return ARG_VALID();
	return ARG_INVALID("Gap penalty cannot be set for non-linear methods");
}

static struct arg_callback validate_gap_affine(void)
{
	if (ALIGN->gap == GAP_AFFINE)
		return ARG_VALID();
	return ARG_INVALID("Affine gaps cannot be set for non-affine methods");
}

static void print_gap_value(void)
{
	if (ALIGN->gap == GAP_LINEAR)
		pinfom("Gap penalty: %d", GAP_PEN);
	else if (ALIGN->gap == GAP_AFFINE)
		pinfom("Gap open: %d, extend: %d", GAP_OPN, GAP_EXT);
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
	.action_callback = print_gap_value,
	.action_order = ARG_ORDER_AFTER(ARG(align)),
	.help_order = ARG_ORDER_AFTER(ARG(align)),
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(align)),
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
	.validate_order = ARG_ORDER_AFTER(ARG(substitution_matrix)),
	.help_order = ARG_ORDER_AFTER(ARG(gap_penalty)),
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(align)),
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
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(align)),
	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(gap_penalty)),
};
