#include "bio/score/matrices.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "util/args.h"
#include "util/print.h"

alignas(CACHE_LINE) s32 SEQ_LUT[SEQ_LUT_SIZE];
alignas(CACHE_LINE) s32 SUB_MAT[SUB_MAT_DIM][SUB_MAT_DIM];

typedef struct {
	const char *name;
	const s32 *data;
	const s32 *lut;
} matrix;

#define GENERATED_MATRICES_IMPLEMENTATION
#include "generated/bio/score/matrices.h"

static const char *selected;

static struct arg_callback parse_matrix(const char *str, void *dest)
{
	(void)dest;
	for (size_t i = 0; i < ARRAY_SIZE(MATRICES); i++) {
		if (strcasecmp(str, MATRICES[i].name) == 0) {
			selected = MATRICES[i].name;
			memcpy(SEQ_LUT, MATRICES[i].lut, sizeof(SEQ_LUT));
			memcpy(SUB_MAT, MATRICES[i].data, sizeof(SUB_MAT));
			return ARG_VALID();
		}
	}
	return ARG_INVALID("Invalid substitution matrix name");
}

static void print_matrix(void)
{
	pinfom("Matrix: %s", selected);
}

static void print_matrix_group(const char *title, int count, int start)
{
	printf("\n%s (%d):\n", title, count);
	for (int i = 0; i < count; i += 5) {
		printf("  ");
		const int next = i + 5;
		const int row_end = next < count ? next : count;
		for (int j = i; j < row_end; j++)
			printf("%-10s", MATRICES[start + j].name);
		putchar('\n');
	}
}

static struct arg_callback list_matrices(const char *str, void *dest)
{
	(void)str;
	(void)dest;
	printf("\nListing available substitution matrices\n");
	print_matrix_group("Amino Matrices", AMINO_MAT_N, 0);
	print_matrix_group("Nucleotide Matrices", NUCLEO_MAT_N, AMINO_MAT_N);
	exit(EXIT_SUCCESS);
}

ARG_EXTERN(output_path);

ARGUMENT(substitution_matrix) = {
	.opt = 'm',
	.lopt = "matrix",
	.help = "Substitution matrix\n  Use -l, --list-matrices to see all available matrices",
	.param = "MATRIX",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.parse_callback = parse_matrix,
	.action_callback = print_matrix,
	.action_order = ARG_ORDER_AFTER(ARG(output_path)),
	.help_order = ARG_ORDER_AFTER(ARG(output_path)),
};

ARGUMENT(list_matrices) = {
	.opt = 'l',
	.lopt = "list-matrices",
	.help = "List available substitution matrices",
	.parse_callback = list_matrices,
	.help_order = ARG_ORDER_FIRST,
};
