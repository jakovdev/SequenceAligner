#include "cuda_manager.cuh"
#include "host_interface.h"
#include "host_types.h"

extern "C" {
bool cuda_initialize()
{
	return Cuda::getInstance().initialize();
}

bool cuda_triangular(size_t buffer_bytes)
{
	return !Cuda::getInstance().hasEnoughMemory(buffer_bytes);
}

bool cuda_upload_sequences(char *sequences_letters, u32 *sequences_offsets,
			   u32 *sequences_lengths, u32 sequences_count,
			   u64 total_sequences_length)
{
	return Cuda::getInstance().uploadSequences(
		sequences_letters, sequences_offsets, sequences_lengths,
		sequences_count, total_sequences_length);
}

bool cuda_upload_scoring(int *sub_matrix, int *sequence_lookup)
{
	return Cuda::getInstance().uploadScoring(sub_matrix, sequence_lookup);
}

bool cuda_upload_gaps(s32 linear, s32 open, s32 extend)
{
	return Cuda::getInstance().uploadGaps(linear, open, extend);
}

bool cuda_upload_indices(u64 *indices, s32 *scores, size_t scores_bytes)
{
	return Cuda::getInstance().uploadIndices(indices, scores, scores_bytes);
}

bool cuda_kernel_launch(int kernel_id)
{
	return Cuda::getInstance().launchKernel(kernel_id);
}

bool cuda_results_get()
{
	return Cuda::getInstance().getResults();
}

ull cuda_results_progress()
{
	return Cuda::getInstance().getProgress();
}

sll cuda_results_checksum()
{
	return Cuda::getInstance().getChecksum();
}

const char *cuda_error_device_get()
{
	return Cuda::getInstance().getDeviceError();
}

const char *cuda_error_host_get()
{
	return Cuda::getInstance().getHostError();
}

const char *cuda_device_name()
{
	return Cuda::getInstance().getDeviceName();
}
}
