#include "io/output.h"

#include <args.h>
#include <print.h>
#include <string.h>

#include "interface/seqalign_cuda.h"
#include "bio/align.h"
#include "io/input.h"
#include "system/os.h"
#include "util/benchmark.h"

static bool disable_write;
static const char *OUTPUT_PATH;

bool output_load(struct output *out, struct input in)
{
	if (disable_write)
		return true;

	const char **MALLOCA(seqs, in.num);
	if (!seqs) {
		perr("Out of memory allocating output sequence data");
		return false;
	}

	for (s32 i = 0; i < in.num; i++)
		seqs[i] = (char *)(in.seqs + in.meta[i].off);

	out->seqs = seqs;
	out->dim = (size_t)in.num;
	size_t bytes = bytesof(out->matrix, in.num * in.num);
	bool tmpf = bytes > available_memory() * 3 / 4;
	out->triangular = tmpf || !cuda_memory(bytes);
	if (out->triangular) {
		bytes = bytesof(out->matrix, alignments(out->dim));
		pverb("Using triangular matrix storage");
	}
	if (tmpf) {
		pinfo("Similarity Matrix size exceeds memory limits");
		pinfol("Creating temporary matrix file (%.2f GiB)",
		       (double)bytes / (double)GiB);
	}

	bench_output_start();
	out->matrix = alloc_mmap(bytes, tmpf);
	if (!out->matrix)
		return false;
	bench_output_end();

	pinfo("Similarity Matrix size: %d x %d", in.num, in.num);
	return true;
}

void output_fill(struct output out, const s32 *cols, size_t col)
{
	if (disable_write)
		return;

	if (!out.matrix || col >= out.dim)
		unreachable_release();

	if (!out.triangular) {
		for (size_t row = 0; row < col; row++) {
			out.matrix[out.dim * row + col] = cols[row];
			out.matrix[out.dim * col + row] = cols[row];
		}
		return;
	}
	memcpy(out.matrix + alignments(col), cols, bytesof(out.matrix, col));
}

bool (*FLUSH_FORMATS[FLUSH_COUNT])(const struct output *, const char *);
enum output_format FLUSH_ID = FLUSH_HDF5 /* FLUSH_INVALID */;

bool output_flush(const struct output *out)
{
	if (disable_write)
		return true;
	bench_output_start();
	bool retval = FLUSH_FORMATS[FLUSH_ID](out, OUTPUT_PATH);
	bench_output_end();
	bench_output_print();
	return retval;
}

void output_free(struct output *out)
{
	free(out->seqs);
	free_mmap(out->matrix);
	memset(out, 0, sizeof(*out));
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
