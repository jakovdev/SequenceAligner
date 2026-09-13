#include "bio/align.h"

#include <args.h>
#include <print.h>
#include <progress.h>
#include <string.h>
#include <strings.h>

#include "bio/method.h"
#include "io/input.h"
#include "io/output.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

shz GAP_PEN;
shz GAP_OPN;
shz GAP_EXT;

usz TABLE_SIZE;
const struct methods *ALIGN;

bool align_cpu(struct input in, struct output out)
{
	usz alignments = alignments((usz)in.num);
	pinfo("Performing %zu pairwise alignments", alignments);
	progress_start(alignments, THREAD_NUM, "Aligning sequences");

	TABLE_SIZE = (usz)(in.max + 1) * (in.max + 1);
	usz mult = ALIGN->gap == GAP_AFFINE ? 3 : 1;
	auto method = ALIGN->method;
	bench_align_start();
#pragma omp parallel
	{
		shz *MALLOCA_AL(s1i, CACHE_LINE, TABLE_SIZE * mult + in.max);
		shz *MALLOCA_AL(cols, CACHE_LINE, in.num);
		if (!s1i || !cols) {
#pragma omp single
			{
				perr("Out of memory for alignment buffers");
				pabort();
			}
		}
#pragma omp for schedule(dynamic)
		for (uhz j = 1; j < in.num; j++) {
			struct meta m1 = in.meta[j];
			const u8 *restrict s1 = in.seqs + m1.off;
			uhz l1 = m1.len;
			for (uhz i = 0; i < l1; ++i)
				s1i[i] = SEQ_LUT[s1[i]];
			for (uhz i = 0; i < j; i++) {
				struct meta m2 = in.meta[i];
				const u8 *restrict s2 = in.seqs + m2.off;
				uhz l2 = m2.len;
				cols[i] = method(l1, l2, s1i, s2);
			}

			output_fill(out, cols, j);
			progress_add(j);
		}

		free_aligned(cols);
		free_aligned(s1i);
	}

	bench_align_end();
	progress_end();
	bench_align_print();
	return true;
}

static struct arg_callback parse_align(const char *str, void *)
{
	for (ALIGN = __start_methods; ALIGN < __stop_methods; ALIGN++) {
		if (strcasecmp(str, ALIGN->arg) == 0 ||
		    strcasecmp(str, ALIGN->name) == 0)
			return ARG_VALID();
	}
	ALIGN = nullptr;
	return ARG_INVALID("Invalid alignment method");
}

static struct arg_callback validate_align(void)
{
	return ALIGN->validate ? ALIGN->validate() : ARG_VALID();
}

static void print_align(void)
{
	pinfom("Method: %s", ALIGN->name);
}

ARG_EXTERN(substitution_matrix);
ARG_EXTERN(gap_penalty);

static char help[512];

[[gnu::constructor]]
static void build_help_strings(void)
{
	snprintf(help, sizeof(help), "Alignment method\n");
	for (auto m = __start_methods; m < __stop_methods; m++) {
		usz len = strlen(help);
		snprintf(help + len, sizeof(help) - len, "  %s: %s\n", m->name,
			 m->arg);
	}
}

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

ARG_PARSE_L(parse_gap_value, 10, shz, -(shz), (val < 0 || val > SHZ_MAX),
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
