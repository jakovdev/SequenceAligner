#include "bio/types.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bio/algorithm/method/ga.h"
#include "bio/algorithm/method/nw.h"
#include "bio/algorithm/method/sw.h"
#include "bio/score/matrices.h"
#include "system/compiler.h"

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

align_func_t align_function(enum AlignmentMethod method)
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
		UNREACHABLE();
	}
}

const char *alignment_name(enum AlignmentMethod method)
{
	return ALIGNMENT_METHODS[method].name;
}

bool alignment_gap_type(enum AlignmentMethod method, enum GapPenaltyType type)
{
	return ALIGNMENT_METHODS[method].gap_type == type;
}

bool alignment_linear(enum AlignmentMethod method)
{
	return alignment_gap_type(method, GAP_TYPE_LINEAR);
}

bool alignment_affine(enum AlignmentMethod method)
{
	return alignment_gap_type(method, GAP_TYPE_AFFINE);
}

/* EXPANDABLE: enum GapPenaltyType */

const char *gap_type_name(enum AlignmentMethod method)
{
	switch (ALIGNMENT_METHODS[method].gap_type) {
	case GAP_TYPE_LINEAR:
		return "Linear";
	case GAP_TYPE_AFFINE:
		return "Affine";
	default:
		UNREACHABLE();
	}
}

enum AlignmentMethod alignment_arg(const char *arg)
{
	if (!arg)
		return ALIGN_INVALID;

	// numeric
	if (isdigit(arg[0]) || (arg[0] == '-' && isdigit(arg[1]))) {
		int method = atoi(arg);
		if (method >= 0 && method < ALIGN_COUNT)
			return (enum AlignmentMethod)method;

		return ALIGN_INVALID;
	}

	// alias
	for (int i = 0; i < ALIGN_COUNT; i++) {
		for (const char **alias = ALIGNMENT_METHODS[i].aliases;
		     *alias != NULL; alias++) {
			if (strcasecmp(arg, *alias) == 0)
				return ALIGNMENT_METHODS[i].method;
		}
	}

	return ALIGN_INVALID;
}

void alignment_list(void)
{
	for (int i = 0; i < ALIGN_COUNT; i++) {
		printf("                           %s: %s (%s, %s gap)\n",
		       ALIGNMENT_METHODS[i].aliases[0],
		       ALIGNMENT_METHODS[i].name,
		       ALIGNMENT_METHODS[i].description,
		       gap_type_name((enum AlignmentMethod)i));
	}
}

const char *matrix_id_name(enum SequenceType seq_type, int matrix_id)
{
	if (seq_type < 0 || matrix_id < 0)
		return "Unknown";

	if (seq_type == SEQ_TYPE_AMINO && matrix_id < NUM_AMINO_MATRICES)
		return AMINO_MATRIX[matrix_id].name;
	if (seq_type == SEQ_TYPE_NUCLEO && matrix_id < NUM_NUCLEO_MATRICES)
		return NUCLEO_MATRIX[matrix_id].name;
	/* EXPANDABLE: enum SequenceType */

	return "Unknown";
}

int matrix_name_id(enum SequenceType seq_type, const char *name)
{
	if (!name)
		return -1;

	int num_matrices = 0;
	const void *matrices = NULL;

	if (seq_type == SEQ_TYPE_AMINO) {
		num_matrices = NUM_AMINO_MATRICES;
		matrices = AMINO_MATRIX;
	} else if (seq_type == SEQ_TYPE_NUCLEO) {
		num_matrices = NUM_NUCLEO_MATRICES;
		matrices = NUCLEO_MATRIX;
	} else /* EXPANDABLE: enum SequenceType */ {
		return -1;
	}

	for (int i = 0; i < num_matrices; i++) {
		const char *matrix_name = NULL;
		if (seq_type == SEQ_TYPE_AMINO)
			matrix_name = ((const AminoMatrix *)matrices)[i].name;
		else if (seq_type == SEQ_TYPE_NUCLEO)
			matrix_name = ((const NucleoMatrix *)matrices)[i].name;
		/* EXPANDABLE: enum SequenceType */

		if (strcasecmp(name, matrix_name) == 0)
			return i;
	}

	return -1;
}

void matrix_seq_type_list(enum SequenceType seq_type)
{
	if (seq_type == SEQ_TYPE_AMINO) {
		for (int i = 0; i < NUM_AMINO_MATRICES; i++)
			printf("  %s%s", AMINO_MATRIX[i].name,
			       (i + 1) % 5 == 0		     ? "\n" :
			       (i == NUM_AMINO_MATRICES - 1) ? "\n" :
							       ", ");
	} else if (seq_type == SEQ_TYPE_NUCLEO) {
		for (int i = 0; i < NUM_NUCLEO_MATRICES; i++)
			printf("  %s%s", NUCLEO_MATRIX[i].name,
			       (i + 1) % 5 == 0		      ? "\n" :
			       (i == NUM_NUCLEO_MATRICES - 1) ? "\n" :
								", ");
	} /* EXPANDABLE: enum SequenceType */
}

const char *sequence_type_name(enum SequenceType seq_type)
{
	return SEQUENCE_TYPES[seq_type].name;
}

enum SequenceType sequence_type_arg(const char *arg)
{
	if (!arg)
		return SEQ_TYPE_INVALID;

	if (isdigit(arg[0]) || (arg[0] == '-' && isdigit(arg[1]))) {
		int type = atoi(arg);
		if (type >= 0 && type < SEQ_TYPE_COUNT)
			return (enum SequenceType)type;

		return SEQ_TYPE_INVALID;
	}

	for (int i = 0; i < SEQ_TYPE_COUNT; i++) {
		for (const char **alias = SEQUENCE_TYPES[i].aliases;
		     *alias != NULL; alias++) {
			if (strcasecmp(arg, *alias) == 0)
				return SEQUENCE_TYPES[i].type;
		}
	}

	return SEQ_TYPE_INVALID;
}

void sequence_types_list(void)
{
	for (int i = 0; i < SEQ_TYPE_COUNT; i++)
		printf("                           %s: %s (%s)\n",
		       SEQUENCE_TYPES[i].aliases[0], SEQUENCE_TYPES[i].name,
		       SEQUENCE_TYPES[i].description);
}
