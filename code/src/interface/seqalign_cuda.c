#include "interface/seqalign_cuda.h"

#include "util/args.h"
#include "util/print.h"

#ifdef USE_CUDA
#include "bio/score/scoring.h"
#include "bio/sequence/sequences.h"
#include "interface/seqalign_hdf5.h"
#include "util/benchmark.h"

#include "host_interface.h"

#define RETURN_CUDA_ERRORS(error_message_lit)                                  \
	do {                                                                   \
		print(M_LOC(FIRST), ERR error_message_lit);                    \
		print(M_LOC(MIDDLE), ERR "Host error: %s", cuda_error_host()); \
		print(M_LOC(LAST), ERR "Device error: %s",                     \
		      cuda_error_device());                                    \
		return false;                                                  \
	} while (0)

bool cuda_align(void)
{
	print_error_context("CUDA");

	const char *device_name = cuda_device_name();
	if (!device_name)
		RETURN_CUDA_ERRORS("Failed to query device name");

	print(M_NONE, INFO "Using CUDA device: %s", device_name);

	if (!cuda_upload_sequences(sequences(), sequences_count(),
				   sequences_length_max(),
				   sequences_length_sum()))
		RETURN_CUDA_ERRORS("Failed uploading sequences");

	if (!cuda_upload_scoring(SUB_MAT, SEQ_LUP))
		RETURN_CUDA_ERRORS("Failed uploading scoring data");

	if (!cuda_upload_gaps(arg_gap_pen(), arg_gap_open(), arg_gap_ext()))
		RETURN_CUDA_ERRORS("Failed uploading gaps");

	if (!cuda_upload_storage(h5_matrix_data(), h5_matrix_bytes()))
		RETURN_CUDA_ERRORS("Failed uploading results storage");

	print(M_PERCENT(0) "Aligning sequences");
	bench_align_start();
	while (true) {
		if (!cuda_kernel_launch(arg_align_method()))
			RETURN_CUDA_ERRORS("Failed to launch alignment");

		if (!cuda_kernel_results())
			RETURN_CUDA_ERRORS("Failed to get results");

		ull progress = cuda_kernel_progress();
		const u64 alignments = sequences_alignment_count();
		print(M_PROPORT(progress / alignments) "Aligning sequences");
		if (progress >= alignments)
			break;
	}

	bench_align_end();
	print(M_PERCENT(100) "Aligning sequences");
	h5_checksum_set(cuda_kernel_checksum() * 2);
	return true;
}

static bool no_cuda;

bool arg_mode_cuda(void)
{
	return !no_cuda;
}

static void print_no_cuda(void)

{
	print(M_LOC(MIDDLE), INFO "CUDA: Enabled");
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
	print(M_LOC(MIDDLE), WARNING "CUDA: Ignored");
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
