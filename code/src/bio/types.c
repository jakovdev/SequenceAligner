#include "bio/types.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "system/compiler.h"
#include "util/args.h"
#include "util/print.h"

/* NOTE: Additional types can be added here if needed.
 *       However, this requires implementing new arguments.
 */
enum GapPenaltyType {
	GAP_TYPE_LINEAR,
	GAP_TYPE_AFFINE,
	/* NOTE: EXPANDABLE enum GapPenaltyType */
};

#define GA_ALIASES ((const char *[]){ "ga", "gotoh", NULL })
#define NW_ALIASES ((const char *[]){ "nw", "needleman", NULL })
#define SW_ALIASES ((const char *[]){ "sw", "smith", NULL })
static struct {
	const char *name;
	const char *description;
	const char **aliases;
	enum AlignmentMethod method;
	enum GapPenaltyType gap_type;
} ALIGNMENT_METHODS[] = {
	{ "Gotoh (affine)", "global alignment", GA_ALIASES, ALIGN_GOTOH_AFFINE,
	  GAP_TYPE_AFFINE },
	{ "Needleman-Wunsch", "global alignment", NW_ALIASES,
	  ALIGN_NEEDLEMAN_WUNSCH, GAP_TYPE_LINEAR },
	{ "Smith-Waterman", "local alignment", SW_ALIASES, ALIGN_SMITH_WATERMAN,
	  GAP_TYPE_AFFINE }
	/* NOTE: EXPANDABLE enum AlignmentMethod, enum GapPenaltyType */
};

s32 GAP_PEN;
s32 GAP_OPEN;
s32 GAP_EXT;

static enum AlignmentMethod method_id = ALIGN_INVALID;
enum AlignmentMethod arg_align_method(void)
{
	return method_id;
}

static struct arg_callback parse_align_method(const char *str, void *dest)
{
	(void)dest;
	method_id = ALIGN_INVALID;
	errno = 0;
	char *endptr = NULL;
	long id = strtol(str, &endptr, 10);
	if (endptr != str && *endptr == '\0' && errno != ERANGE &&
	    id > ALIGN_INVALID && id < ALIGN_COUNT)
		method_id = (enum AlignmentMethod)id;

	if (method_id == ALIGN_INVALID) {
		for (int i = 0; i < ALIGN_COUNT; i++) {
			for (const char **alias = ALIGNMENT_METHODS[i].aliases;
			     *alias != NULL; alias++) {
				if (strcasecmp(str, *alias) == 0)
					method_id = ALIGNMENT_METHODS[i].method;
			}
		}
	}

	if (method_id == ALIGN_INVALID)
		return ARG_INVALID("Invalid alignment method");

	return ARG_VALID();
}

ARG_PARSE_L(gap_value, 10, s32, -(s32), (val < 0 || val > INT_MAX),
	    "Gap values must be positive integers")

static struct arg_callback validate_gap_pen(void)
{
	if (ALIGNMENT_METHODS[method_id].gap_type != GAP_TYPE_LINEAR)
		return ARG_INVALID(
			"Gap penalty cannot be set for non-linear methods");

	return ARG_VALID();
}

static struct arg_callback validate_gap_affine(void)
{
	if (ALIGNMENT_METHODS[method_id].gap_type != GAP_TYPE_AFFINE)
		return ARG_INVALID(
			"Gap open/extend cannot be set for non-affine methods");

	if (method_id == ALIGN_GOTOH_AFFINE && GAP_OPEN == GAP_EXT) {
		if (print_Yn(
			    "Equal gap penalties found, switch to Needleman-Wunsch?")) {
			method_id = ALIGN_NEEDLEMAN_WUNSCH;
			GAP_PEN = GAP_OPEN;
			GAP_OPEN = INT32_MIN;
			GAP_EXT = INT32_MIN;
		}
	}

	return ARG_VALID();
}
static void print_config_method(void)
{
	pinfom("Method: %s", ALIGNMENT_METHODS[method_id].name);
}

static void print_config_gaps(void)
{
	if (ALIGNMENT_METHODS[method_id].gap_type == GAP_TYPE_LINEAR)
		pinfom("Gap penalty: " Ps32, GAP_PEN);
	else if (ALIGNMENT_METHODS[method_id].gap_type == GAP_TYPE_AFFINE)
		pinfom("Gap open: " Ps32 ", extend: " Ps32, GAP_OPEN, GAP_EXT);
	else /* NOTE: EXPANDABLE enum GapPenaltyType */
		unreachable();
}

static char align_help[512];

static const char *gap_type_name(enum AlignmentMethod method)
{
	switch (ALIGNMENT_METHODS[method].gap_type) {
	case GAP_TYPE_LINEAR:
		return "Linear";
	case GAP_TYPE_AFFINE:
		return "Affine";
	default: /* NOTE: EXPANDABLE enum GapPenaltyType */
		unreachable();
	}
}

_ARGS_CONSTRUCTOR(build_help_strings)
{
	snprintf(align_help, sizeof(align_help), "Alignment method\n");
	for (int i = 0; i < ALIGN_COUNT; i++) {
		const char *newline = (i == ALIGN_COUNT - 1) ? "" : "\n";
		size_t len = strlen(align_help);
		snprintf(align_help + len, sizeof(align_help) - len,
			 "  %s: %s (%s, %s gap)%s",
			 ALIGNMENT_METHODS[i].aliases[0],
			 ALIGNMENT_METHODS[i].name,
			 ALIGNMENT_METHODS[i].description, gap_type_name(i),
			 newline);
	}
}

ARG_EXTERN(substitution_matrix);

ARGUMENT(alignment_method) = {
	.opt = 'a',
	.lopt = "align",
	.help = align_help,
	.param = "METHOD",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.parse_callback = parse_align_method,
	.action_callback = print_config_method,
	.action_order = ARG_ORDER_AFTER(ARG(substitution_matrix)),
	.help_order = ARG_ORDER_AFTER(ARG(substitution_matrix)),
};

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
	.dest = &GAP_OPEN,
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
