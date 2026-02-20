#include "bio/types.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bio/algorithm/method/ga.h"
#include "bio/algorithm/method/nw.h"
#include "bio/algorithm/method/sw.h"
#include "system/compiler.h"
#include "util/args.h"
#include "util/print.h"

#define AMINO_ALIASES ((const char *[]){ "amino", "aa", "protein", NULL })
#define NUCLEO_ALIASES ((const char *[]){ "nucleo", "dna", "rna", "nt", NULL })
static struct {
	const char *name;
	const char *description;
	const char **aliases;
	enum SequenceType type;
} SEQUENCE_TYPES[] = {
	{ "Amino acids", "Protein sequences", AMINO_ALIASES, SEQ_TYPE_AMINO },
	{ "Nucleotides", "DNA/RNA sequences", NUCLEO_ALIASES, SEQ_TYPE_NUCLEO },
	/* NOTE: EXPANDABLE enum SequenceType */
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

align_func_t align_method(enum AlignmentMethod method)
{
	switch (method) {
	case ALIGN_GOTOH_AFFINE:
		return align_ga;
	case ALIGN_NEEDLEMAN_WUNSCH:
		return align_nw;
	case ALIGN_SMITH_WATERMAN:
		return align_sw;
	case ALIGN_INVALID:
	case ALIGN_COUNT:
	default: /* NOTE: EXPANDABLE enum AlignmentMethod */
		pdev("Invalid AlignmentMethod enum");
		perr("Internal error retrieving alignment method");
		pabort();
	}
}

static enum SequenceType seq_type = SEQ_TYPE_INVALID;
static int matrix_id = -1;

static enum AlignmentMethod method_id = ALIGN_INVALID;
static s32 gap_pen;
static s32 gap_open;
static s32 gap_ext;

s32 SEQ_LUP[SCHAR_MAX + 1];
s32 SUB_MAT[SUBMAT_MAX][SUBMAT_MAX];

enum AlignmentMethod arg_align_method(void)
{
	return method_id;
}
s32 arg_gap_pen(void)
{
	return gap_pen;
}
s32 arg_gap_open(void)
{
	return gap_open;
}
s32 arg_gap_ext(void)
{
	return gap_ext;
}
enum SequenceType arg_sequence_type(void)
{
	return seq_type;
}
int arg_sub_matrix(void)
{
	return matrix_id;
}

static struct arg_callback parse_seq_type(const char *str, void *dest)
{
	enum SequenceType type = SEQ_TYPE_INVALID;
	errno = 0;
	char *endptr = NULL;
	long id = strtol(str, &endptr, 10);
	if (endptr != str && *endptr == '\0' && errno != ERANGE &&
	    id > SEQ_TYPE_INVALID && id < SEQ_TYPE_COUNT)
		type = (enum SequenceType)id;

	for (int i = 0; i < SEQ_TYPE_COUNT; i++) {
		for (const char **alias = SEQUENCE_TYPES[i].aliases;
		     *alias != NULL; alias++) {
			if (strcasecmp(str, *alias) == 0)
				type = SEQUENCE_TYPES[i].type;
		}
	}

	if (type == SEQ_TYPE_INVALID)
		return ARG_INVALID("Invalid sequence type");

	*(enum SequenceType *)dest = type;
	return ARG_VALID();
}

static struct arg_callback parse_matrix(const char *str, void *dest)
{
	int id = -1;
	switch (seq_type) {
	case SEQ_TYPE_AMINO:
		for (int i = 0; i < NUM_AMINO_MATRICES; i++) {
			if (strcasecmp(str, AMINO_MATRIX[i].name) == 0) {
				id = i;
				break;
			}
		}
		break;
	case SEQ_TYPE_NUCLEO:
		for (int i = 0; i < NUM_NUCLEO_MATRICES; i++) {
			if (strcasecmp(str, NUCLEO_MATRIX[i].name) == 0) {
				id = i;
				break;
			}
		}
		break;
	case SEQ_TYPE_INVALID:
	case SEQ_TYPE_COUNT:
	default: /* NOTE: EXPANDABLE enum SequenceType */
		unreachable();
	}

	if (id < 0)
		return ARG_INVALID("Invalid substitution matrix name");

	*(int *)dest = id;
	return ARG_VALID();
}

static struct arg_callback parse_align_method(const char *str, void *dest)
{
	enum AlignmentMethod method = ALIGN_INVALID;
	errno = 0;
	char *endptr = NULL;
	long id = strtol(str, &endptr, 10);
	if (endptr != str && *endptr == '\0' && errno != ERANGE &&
	    id > ALIGN_INVALID && id < ALIGN_COUNT)
		method = (enum AlignmentMethod)id;

	if (method == ALIGN_INVALID) {
		for (int i = 0; i < ALIGN_COUNT; i++) {
			for (const char **alias = ALIGNMENT_METHODS[i].aliases;
			     *alias != NULL; alias++) {
				if (strcasecmp(str, *alias) == 0)
					method = ALIGNMENT_METHODS[i].method;
			}
		}
	}

	if (method == ALIGN_INVALID)
		return ARG_INVALID("Invalid alignment method");

	*(enum AlignmentMethod *)dest = method;
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

	if (method_id == ALIGN_GOTOH_AFFINE && gap_open == gap_ext) {
		if (print_Yn(
			    "Equal gap penalties found, switch to Needleman-Wunsch?")) {
			method_id = ALIGN_NEEDLEMAN_WUNSCH;
			gap_pen = gap_open;
			gap_open = INT32_MIN;
			gap_ext = INT32_MIN;
		}
	}

	return ARG_VALID();
}

static void print_config_seq_type(void)
{
	pinfom("Sequence type: %s", SEQUENCE_TYPES[seq_type].name);
}

static void setup_matrix(void)
{
	memset(SEQ_LUP, -1, sizeof(SEQ_LUP));
	const char *name = "Unknown";

#define SEQ_TYPE_INIT(TYPE)                                               \
	for (int i = 0; i < TYPE##_SIZE; i++)                             \
		SEQ_LUP[(uchar)TYPE##_ALPHABET[i]] = i;                   \
	memcpy(SUB_MAT, TYPE##_MATRIX[matrix_id].matrix, TYPE##_MATSIZE); \
	name = TYPE##_MATRIX[matrix_id].name

	switch (seq_type) {
	case SEQ_TYPE_AMINO:
		SEQ_TYPE_INIT(AMINO);
		break;
	case SEQ_TYPE_NUCLEO:
		SEQ_TYPE_INIT(NUCLEO);
		break;
	case SEQ_TYPE_INVALID:
	case SEQ_TYPE_COUNT:
	default: /* NOTE: EXPANDABLE enum SequenceType */
		unreachable();
	}
#undef SEQ_TYPE_INIT
	pinfom("Matrix: %s", name);
}

static void print_config_method(void)
{
	pinfom("Method: %s", ALIGNMENT_METHODS[method_id].name);
}

static void print_config_gaps(void)
{
	if (ALIGNMENT_METHODS[method_id].gap_type == GAP_TYPE_LINEAR)
		pinfom("Gap penalty: " Ps32, gap_pen);
	else if (ALIGNMENT_METHODS[method_id].gap_type == GAP_TYPE_AFFINE)
		pinfom("Gap open: " Ps32 ", extend: " Ps32, gap_open, gap_ext);
	else /* NOTE: EXPANDABLE enum GapPenaltyType */
		unreachable();
}

static char seq_type_help[512];
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
	snprintf(seq_type_help, sizeof(seq_type_help), "Sequence type\n");
	for (int i = 0; i < SEQ_TYPE_COUNT; i++) {
		const char *newline = (i == SEQ_TYPE_COUNT - 1) ? "" : "\n";
		size_t len = strlen(seq_type_help);
		snprintf(seq_type_help + len, sizeof(seq_type_help) - len,
			 "  %s: %s%s", SEQUENCE_TYPES[i].aliases[0],
			 SEQUENCE_TYPES[i].name, newline);
	}

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

static struct arg_callback list_matrices(const char *str, void *dest)
{
	(void)str;
	(void)dest;
	printf("Listing available substitution matrices\n\n");

	printf("Amino Acid Matrices (%d):\n", NUM_AMINO_MATRICES);
	for (int i = 0; i < NUM_AMINO_MATRICES; i++) {
		const char *sep = ((i + 1) % 5 == 0)		? "\n" :
				  (i == NUM_AMINO_MATRICES - 1) ? "\n" :
								  ", ";
		printf("  %s%s", AMINO_MATRIX[i].name, sep);
	}

	printf("\nNucleotide Matrices (%d):\n", NUM_NUCLEO_MATRICES);
	for (int i = 0; i < NUM_NUCLEO_MATRICES; i++) {
		const char *sep = ((i + 1) % 5 == 0)		 ? "\n" :
				  (i == NUM_NUCLEO_MATRICES - 1) ? "\n" :
								   ", ";
		printf("  %s%s", NUCLEO_MATRIX[i].name, sep);
	}

	/* NOTE: EXPANDABLE enum SequenceType */

	exit(EXIT_SUCCESS);
}

ARG_EXTERN(output_path);

ARGUMENT(sequence_type) = {
	.opt = 't',
	.lopt = "type",
	.help = seq_type_help,
	.param = "TYPE",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = &seq_type,
	.parse_callback = parse_seq_type,
	.action_callback = print_config_seq_type,
	.action_order = ARG_ORDER_AFTER(output_path),
	.help_order = ARG_ORDER_AFTER(output_path),
};

ARGUMENT(substitution_matrix) = {
	.opt = 'm',
	.lopt = "matrix",
	.help = "Substitution matrix\n  Use -l, --list-matrices to see all available matrices",
	.param = "MATRIX",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = &matrix_id,
	.parse_callback = parse_matrix,
	.action_callback = setup_matrix,
	.action_order = ARG_ORDER_AFTER(sequence_type),
	.help_order = ARG_ORDER_AFTER(sequence_type),
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(sequence_type)),
};

ARGUMENT(alignment_method) = {
	.opt = 'a',
	.lopt = "align",
	.help = align_help,
	.param = "METHOD",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = &method_id,
	.parse_callback = parse_align_method,
	.action_callback = print_config_method,
	.action_order = ARG_ORDER_AFTER(substitution_matrix),
	.help_order = ARG_ORDER_AFTER(substitution_matrix),
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
	.dest = &gap_pen,
	.parse_callback = parse_gap_value,
	.validate_callback = validate_gap_pen,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_AFTER(gap_open),
	.action_callback = print_config_gaps,
	.action_order = ARG_ORDER_AFTER(alignment_method),
	.help_order = ARG_ORDER_AFTER(alignment_method),
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
	.dest = &gap_open,
	.parse_callback = parse_gap_value,
	.validate_callback = validate_gap_affine,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_AFTER(alignment_method),
	.help_order = ARG_ORDER_AFTER(gap_penalty),
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
	.dest = &gap_ext,
	.parse_callback = parse_gap_value,
	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(alignment_method)),
	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(gap_penalty)),
};

ARGUMENT(list_matrices) = {
	.opt = 'l',
	.lopt = "list-matrices",
	.help = "List available substitution matrices",
	.parse_callback = list_matrices,
	.help_order = ARG_ORDER_FIRST,
};
