#include "io/output.h"

#include <args.h>
#include <print.h>
#include <string.h>

#include "bio/sequence.h"
#include "interface/seqalign_cuda.h"
#include "io/input.h"
#include "system/os.h"
#include "util/benchmark.h"

static bool disable_write;
static const char *OUTPUT_PATH;

bool output_load(struct output *sm, const struct input *in)
{
	if (disable_write)
		return true;

	MALLOCA(sm->seqs, in->num);
	if (!sm->seqs) {
		perr("Out of memory allocating output sequence data");
		return false;
	}

	for (s32 i = 0; i < in->num; i++)
		sm->seqs[i] = in->seqs[i].letters;

	sm->triangular = false;
	sm->dim = (size_t)in->num;
	size_t bytes = bytesof(sm->matrix, sm->dim * sm->dim);
	sm->mmap = bytes > (available_memory() * 3 / 4);
	if (sm->mmap || !cuda_memory(bytes)) {
		bytes = bytesof(sm->matrix, sm->dim * (sm->dim - 1) / 2);
		sm->triangular = true;
		pverb("Using triangular matrix storage");
	}

	bench_output_start();
	if (sm->mmap) {
		pinfo("Similarity Matrix size exceeds memory limits");
		pinfol("Creating temporary matrix file (%.2f GiB)",
		       (double)bytes / (double)GiB);
		sm->matrix = alloc_mmap(bytes);
		if (!sm->matrix)
			return false;
	} else {
		MALLOC_AL(sm->matrix, PAGE_SIZE, bytes);
		if unlikely (!sm->matrix) {
			perr("Out of memory allocating Similarity Matrix");
			return false;
		}
		memset(sm->matrix, 0, bytes);
	}
	bench_output_end();

	pinfo("Similarity Matrix size: %zu x %zu", sm->dim, sm->dim);
	return true;
}

void output_fill(const struct output *sm, const s32 *columns, size_t col)
{
	if (disable_write)
		return;

	if (!sm->matrix || col >= sm->dim)
		unreachable_release();

	if (sm->triangular) {
		memcpy(sm->matrix + col * (col - 1) / 2, columns,
		       bytesof(sm->matrix, col));
	} else {
		for (size_t row = 0; row < col; row++) {
			sm->matrix[sm->dim * row + col] = columns[row];
			sm->matrix[sm->dim * col + row] = columns[row];
		}
	}
}

bool (*FLUSH_FORMATS[FLUSH_COUNT])(const struct output *, const char *);
enum output_format FLUSH_ID = FLUSH_HDF5 /* FLUSH_INVALID */;

bool output_flush(const struct output *sm)
{
	if (disable_write)
		return true;
	bench_output_start();
	bool retval = FLUSH_FORMATS[FLUSH_ID](sm, OUTPUT_PATH);
	bench_output_end();
	bench_output_print();
	return retval;
}

void output_free(struct output *sm)
{
	free(sm->seqs);
	if (sm->mmap)
		free_mmap(sm->matrix);
	else
		free_aligned(sm->matrix);
	memset(sm, 0, sizeof(*sm));
}

ARG_EXTERN(disable_cuda);

ARGUMENT(disable_write) = {
	.opt = 'W',
	.lopt = "no-write",
	.help = "Disable writing to output file",
	.set = &disable_write,
	.help_order = ARG_ORDER_AFTER(ARG(disable_cuda)),
};

static void print_output_path(void)
{
	if (disable_write)
		pwarnm("Output: Ignored");
	else
		pinfom("Output: %s", file_name(OUTPUT_PATH));
}

static struct arg_callback validate_output_path(void)
{
	if (disable_write)
		return ARG_VALID();

	if (path_file_exists(OUTPUT_PATH)) {
		pwarn("Output file already exists: %s", file_name(OUTPUT_PATH));
		if (!print_yN("Do you want to DELETE it?"))
			return ARG_INVALID(
				"Output file exists and will not be overwritten");
		if (remove(OUTPUT_PATH) != 0)
			return ARG_INVALID(
				"Failed to delete existing output file");
		pinfo("Deleted existing output file");
	}

	if (!path_directories_create(OUTPUT_PATH))
		return ARG_INVALID(
			"Failed to create directories for output file");

	return ARG_VALID();
}

ARG_EXTERN(input_path);

ARGUMENT(output_path) = {
	.opt = 'o',
	.lopt = "output",
	.help = "Output file path: HDF5 format",
	.param = "FILE",
	.param_req = ARG_PARAM_REQUIRED,
	.arg_req = ARG_REQUIRED,
	.dest = &OUTPUT_PATH,
	.parse_callback = parse_path,
	.validate_callback = validate_output_path,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.validate_order = ARG_ORDER_AFTER(ARG(input_path)),
	.action_callback = print_output_path,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(input_path)),
	.help_order = ARG_ORDER_AFTER(ARG(input_path)),
	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(disable_write)),
};
