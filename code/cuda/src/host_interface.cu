#include "host_interface.h"

#include "cuda_manager.cuh"

extern "C" {
bool cuda_triangular(size_t buffer_bytes)
{
	return !Cuda::Instance().memoryCheck(buffer_bytes);
}

bool cuda_upload_sequences(const sequence_t *seqs, u32 seq_n, u32 seq_len_max,
			   u64 seq_len_sum)
{
	return Cuda::Instance().uploadSequences(seqs, seq_n, seq_len_max,
						seq_len_sum);
}

bool cuda_upload_scoring(s32 sub_mat[SUB_MATDIM][SUB_MATDIM],
			 s32 seq_lup[SEQ_LUPSIZ])
{
	return Cuda::Instance().uploadScoring(sub_mat, seq_lup);
}

bool cuda_upload_gaps(s32 linear, s32 open, s32 extend)
{
	return Cuda::Instance().uploadGaps(linear, open, extend);
}

bool cuda_upload_storage(s32 *scores, size_t scores_bytes)
{
	return Cuda::Instance().uploadStorage(scores, scores_bytes);
}

bool cuda_kernel_launch(int kernel_id)
{
	return Cuda::Instance().kernelLaunch(kernel_id);
}

bool cuda_kernel_results(void)
{
	return Cuda::Instance().kernelResults();
}

ull cuda_kernel_progress(void)
{
	return Cuda::Instance().kernelProgress();
}

sll cuda_kernel_checksum(void)
{
	return Cuda::Instance().kernelChecksum();
}

const char *cuda_error_device(void)
{
	return Cuda::Instance().deviceError();
}

const char *cuda_error_host(void)
{
	return Cuda::Instance().hostError();
}

const char *cuda_device_name(void)
{
	return Cuda::Instance().deviceName();
}
}
