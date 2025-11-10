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

bool cuda_upload_sequences(const sequence_t *seqs, u32 seq_n, u64 seq_len_sum)
{
	return Cuda::getInstance().uploadSequences(seqs, seq_n, seq_len_sum);
}

bool cuda_upload_scoring(const s32 sub_mat[SUB_MATDIM][SUB_MATDIM],
			 const s32 seq_lup[SEQ_LUPSIZ])
{
	return Cuda::getInstance().uploadScoring(sub_mat, seq_lup);
}

bool cuda_upload_gaps(s32 linear, s32 open, s32 extend)
{
	return Cuda::getInstance().uploadGaps(linear, open, extend);
}

bool cuda_upload_storage(s32 *scores, size_t scores_bytes)
{
	return Cuda::getInstance().uploadStorage(scores, scores_bytes);
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
