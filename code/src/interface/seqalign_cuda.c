#include "interface/seqalign_cuda.h"

#include "util/args.h"
#include "util/print.h"

#ifdef USE_CUDA
#include <string.h>

#include "bio/sequence/sequences.h"
#include "bio/types.h"
#include "interface/seqalign_hdf5.h"
#include "system/compiler.h"
#include "util/benchmark.h"

#include "host_interface.h"

#define RETURN_CUDA_ERRORS(...)                                 \
	do {                                                    \
		perr(__VA_ARGS__);                              \
		perrm("Host error: %s", cuda_error_host());     \
		perrl("Device error: %s", cuda_error_device()); \
		return false;                                   \
	} while (0)

bool cuda_align(void)
{
	const char *name = cuda_device_name();
	if (!name || !name[0] || strcmp(name, "Unknown Device") == 0)
		RETURN_CUDA_ERRORS("Failed to query device name");

	pinfo("Using CUDA device: %s", name);

	if unlikely (!cuda_upload_seqs(sequences_seqs(), sequences_seq_n(),
				       sequences_seq_len_max(),
				       sequences_seq_len_sum()))
		RETURN_CUDA_ERRORS("Failed uploading sequences to device");

	if unlikely (!cuda_upload_scoring(SUB_MAT, SEQ_LUP))
		RETURN_CUDA_ERRORS("Failed uploading scoring data to device");

	if unlikely (!cuda_upload_gaps(arg_gap_pen(), arg_gap_open(),
				       arg_gap_ext()))
		RETURN_CUDA_ERRORS("Failed uploading gaps to device");

	if unlikely (!cuda_upload_storage(h5_matrix_data(), h5_matrix_bytes()))
		RETURN_CUDA_ERRORS("Failed uploading storage data to device");

	const s64 alignments = sequences_alignments();
	pinfol("Performing " Ps64 " pairwise alignments", alignments);

	ppercent(0, "Aligning sequences");
	bench_align_start();
	while (true) {
		if unlikely (!cuda_kernel_launch(arg_align_method()))
			RETURN_CUDA_ERRORS("Failed to launch device alignment");

		if unlikely (!cuda_kernel_results())
			RETURN_CUDA_ERRORS("Failed to get results from device");

		sll progress = cuda_kernel_progress();
		pproportc(progress / alignments, "Aligning sequences");
		if (progress >= alignments)
			break;
	}

	bench_align_end();
	ppercent(100, "Aligning sequences");
	h5_checksum_set(cuda_kernel_checksum() * 2);
	bench_align_print();
	return true;
}

static bool no_cuda;

bool arg_mode_cuda(void)
{
	return !no_cuda;
}

static void print_no_cuda(void)

{
	pinfom("CUDA: Enabled");
}

ARGUMENT(disable_cuda) = {
	.opt = 'C',
	.lopt = "no-cuda",
	.help = "Disable CUDA",
	.set = &no_cuda,
	.action_callback = print_no_cuda,
	.action_phase = ARG_CALLBACK_IF_UNSET,
	.action_weight = 400,
	.help_weight = 350,
};

#undef RETURN_CUDA_ERRORS

#else

bool cuda_align(void)
{
	return false;
}

bool arg_mode_cuda(void)
{
	return false;
}

static void print_cuda_ignored(void)
{
	pwarnm("CUDA: Ignored");
}

ARGUMENT(disable_cuda) = {
	.opt = 'C',
	.lopt = "no-cuda",
	.help = "Disable CUDA (ignored, not compiled with CUDA)",
	.action_callback = print_cuda_ignored,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_weight = 400,
	.help_weight = 350,
};

#endif /* USE_CUDA */
