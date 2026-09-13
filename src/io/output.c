#include "io/output.h"

#include <args.h>
#include <print.h>
#include <string.h>

#include "io/input.h"
#include "io/writer.h"
#include "system/os.h"
#include "util/benchmark.h"

static bool disable_write;
static const char *OUTPUT_PATH;

bool output_load(struct output *out, struct input in)
{
	if (disable_write)
		return true;

	psection("Preparing Similarity Matrix");
	pverb("Using %u sequences for output", in.num);

	pinfo("Similarity Matrix dimensions: %u x %u", in.num, in.num);
	usz bytes = bytesof(out->matrix, in.num * in.num);
	bool tmpf = bytes > memory_cpu() * 3 / 4;
	usz mem_gpu = memory_gpu() * 3 / 4;
	bool triangular = tmpf || mem_gpu ? bytes > mem_gpu : false;
	if (triangular) {
		bytes = bytesof(out->matrix, alignments((usz)in.num));
		pinfo("Using triangular matrix instead of full matrix");
	}
	double usage = (double)bytes / (double)MiB;
	const char *unit = "MiB";
	if (bytes > GiB / 100) {
		usage = (double)bytes / (double)GiB;
		unit = "GiB";
	}
	pinfo("Similarity Matrix size: %.2f %s", usage, unit);
	if (tmpf) {
		pinfom("Similarity Matrix size exceeds memory limits");
		pinfol("Creating temporary file storage: %.2f %s", usage, unit);
	}

	bench_output_start();
	out->matrix = alloc_mmap(bytes, tmpf);
	if (!out->matrix)
		return false;
	bench_output_end();

	out->dim = in.num;
	out->triangular = triangular;
	return true;
}

void output_fill(struct output out, const shz *cols, usz col)
{
	if (disable_write)
		return;

	if (!out.matrix || col >= out.dim)
		unreachable_release();

	if (!out.triangular) {
		for (usz row = 0; row < col; row++) {
			out.matrix[row * out.dim + col] = cols[row];
			out.matrix[col * out.dim + row] = cols[row];
		}
		return;
	}
	memcpy(out.matrix + alignments(col), cols, bytesof(out.matrix, col));
}

bool output_flush(struct output out, struct input in)
{
	if (disable_write)
		return true;

	psection("Writing Similarity Matrix");
	pverb("Trying out writers for %s", file_name(OUTPUT_PATH));
	for (auto s = __start_writers; s < __stop_writers; s++) {
		bench_output_start();
		switch (s->write(out, in, OUTPUT_PATH)) {
		case WRITER_UNSUPPORTED:
			continue;
		case WRITER_SUCCESS:
			bench_output_end();
			bench_output_print();
			return true;
		case WRITER_ERROR:
			return false;
		}
	}

	perr("Unsupported output file format: %s", file_name(OUTPUT_PATH));
	return false;
}

void output_free(struct output *out)
{
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
