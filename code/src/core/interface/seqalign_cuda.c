#include "core/interface/seqalign_cuda.h"

#ifdef USE_CUDA
#include "core/app/args.h"
#include "core/bio/score/scoring.h"
#include "core/bio/sequence/sequences.h"
#include "core/interface/seqalign_hdf5.h"
#include "util/benchmark.h"
#include "util/print.h"

#include "host_interface.h"

#define RETURN_CUDA_ERRORS(error_message_lit)               \
	do {                                                \
		print(M_LOC(FIRST), ERR error_message_lit); \
		print(M_LOC(MIDDLE), ERR "Host error: %s",  \
		      cuda_error_host_get());               \
		print(M_LOC(LAST), ERR "Device error: %s",  \
		      cuda_error_device_get());             \
		return false;                               \
	} while (0)

bool cuda_init(void)
{
	print_error_context("CUDA");

	if (sequences_length_max() > 1024)
		RETURN_CUDA_ERRORS("Sequence length exceeds maximum of 1024");

	if (!cuda_initialize())
		RETURN_CUDA_ERRORS("Failed to create context");

	const char *device_name = cuda_device_name();
	if (!device_name)
		RETURN_CUDA_ERRORS("Failed to query device name");

	print(M_NONE, INFO "Using CUDA device: %s", device_name);
	return true;
}

bool cuda_align(void)
{
	char *seqs = sequences_flattened();
	u32 *offsets = sequences_offsets();
	u32 *lengths = sequences_lengths();
	u32 seqs_n = sequences_count();
	u64 len_total = sequences_length_total();

	print_error_context("CUDA");

	if (!cuda_upload_sequences(seqs, offsets, lengths, seqs_n, len_total))
		RETURN_CUDA_ERRORS("Failed uploading sequences");

	int submat_f[SUBMAT_MAX * SUBMAT_MAX] = { 0 };
	for (int i = 0; i < SUBMAT_MAX; i++) {
		for (int j = 0; j < SUBMAT_MAX; j++)
			submat_f[i * SUBMAT_MAX + j] = SUB_MAT[i][j];
	}

	if (!cuda_upload_scoring(submat_f, SEQ_LUP))
		RETURN_CUDA_ERRORS("Failed uploading scoring data");

	if (!cuda_upload_gaps(args_gap_penalty(), args_gap_open(),
			      args_gap_extend()))
		RETURN_CUDA_ERRORS("Failed uploading gaps");

	s32 *matrix = h5_matrix_data();
	size_t matrix_bytes = h5_matrix_bytes();

	u64 *result_offsets = h5_triangle_indices();
	if (!cuda_upload_indices(result_offsets, matrix, matrix_bytes))
		RETURN_CUDA_ERRORS("Failed uploading results storage");

	bench_align_start();

	if (!cuda_kernel_launch(args_align_method()))
		RETURN_CUDA_ERRORS("Failed to launch alignment");

	const u64 alignments = sequences_alignment_count();

	print(M_PERCENT(0) "Aligning sequences");

	while (true) {
		if (!cuda_results_get())
			RETURN_CUDA_ERRORS("Failed to get results");

		ull progress = cuda_results_progress();
		print(M_PROPORT(progress / alignments) "Aligning sequences");

		if (progress >= alignments)
			break;
		else if (!cuda_kernel_launch(args_align_method()))
			RETURN_CUDA_ERRORS("Failed to launch next batch");
	}

	if (!cuda_results_get())
		RETURN_CUDA_ERRORS("Failed to get results");

	print(M_PERCENT(100) "Aligning sequences");

	h5_checksum_set(cuda_results_checksum() * 2);

	bench_align_end();

	return true;
}

#undef RETURN_CUDA_ERRORS

#else

bool cuda_init(void)
{
	return false;
}

bool cuda_align(void)
{
	return false;
}

#endif // USE_CUDA
