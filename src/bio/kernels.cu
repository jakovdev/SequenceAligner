#include "bio/kernels.cuh"
#include "util/macros.h"

__constant__ struct constants C;
extern "C" const void *const pC = &C;

__forceinline__ __device__ s32 d_seq_lut(s32 ij, s32 pos)
{
	return C.seq_lut[(uchar)C.letters[C.meta[ij].off + pos]];
}

__forceinline__ __device__ s32 d_sub_mat(s32 c1, s32 c2)
{
	return C.sub_mat[c1 * SUB_MAT_DIM + c2];
}

__forceinline__ __device__ s32 d_find_j(s64 alignment)
{
	s32 low = 1, high = C.num;

	while (low < high) {
		s32 mid = low + (high - low) / 2;
		if (alignments(static_cast<s64>(mid)) <= alignment)
			low = mid + 1;
		else
			high = mid;
	}

	return low - 1;
}

extern "C" __global__ void kernel_nw(s32 *scores, s64 start, s64 batch)
{
	s64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	s64 alignment = start + tid;
	s32 j = d_find_j(alignment);
	s32 i = static_cast<s32>(alignment - alignments(static_cast<s64>(j)));

	s32 len1 = C.meta[i].len;
	s32 len2 = C.meta[j].len;

	s32 dp_prev[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 dp_curr[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (s32 col = 0; col <= len2; col++)
		dp_prev[col] = col * C.gap_pen;
	for (s32 row = 1; row <= len1; ++row) {
		dp_curr[0] = row * C.gap_pen;

		for (s32 col = 1; col <= len2; col++) {
			s32 c1 = d_seq_lut(i, row - 1);
			s32 c2 = d_seq_lut(j, col - 1);
			s32 match = dp_prev[col - 1] + d_sub_mat(c1, c2);
			s32 gap_v = dp_prev[col] + C.gap_pen;
			s32 gap_h = dp_curr[col - 1] + C.gap_pen;

			s32 val_max = max(match, gap_v);
			val_max = max(val_max, gap_h);
			dp_curr[col] = val_max;
		}

		for (s32 col = 0; col <= len2; col++)
			dp_prev[col] = dp_curr[col];
	}

	s32 score = dp_prev[len2];
	if (!C.triangular) {
		scores[static_cast<s64>(C.num) * i + j] = score;
		scores[static_cast<s64>(C.num) * j + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(C.progress, 1);
}

extern "C" __global__ void kernel_ga(s32 *scores, s64 start, s64 batch)
{
	s64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	s64 alignment = start + tid;
	s32 j = d_find_j(alignment);
	s32 i = static_cast<s32>(alignment - alignments(static_cast<s64>(j)));

	s32 len1 = C.meta[i].len;
	s32 len2 = C.meta[j].len;

	s32 match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;
	for (s32 col = 1; col <= len2; col++) {
		gap_x[col] = max(match[col - 1] + C.gap_open,
				 gap_x[col - 1] + C.gap_ext);
		match[col] = gap_x[col];
		gap_y[col] = SCORE_MIN;
	}

	s32 p_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 p_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (s32 col = 0; col <= len2; col++) {
		p_match[col] = match[col];
		p_gap_y[col] = gap_y[col];
	}

	for (s32 row = 1; row <= len1; ++row) {
		gap_x[0] = SCORE_MIN;
		gap_y[0] = max(p_match[0] + C.gap_open, p_gap_y[0] + C.gap_ext);
		match[0] = gap_y[0];

		s32 c1 = d_seq_lut(i, row - 1);
		for (s32 col = 1; col <= len2; col++) {
			s32 c2 = d_seq_lut(j, col - 1);
			s32 similarity = d_sub_mat(c1, c2);

			s32 d_score = p_match[col - 1] + similarity;

			s32 open_x = match[col - 1] + C.gap_open;
			s32 extend_x = gap_x[col - 1] + C.gap_ext;
			gap_x[col] = max(open_x, extend_x);

			s32 open_y = p_match[col] + C.gap_open;
			s32 extend_y = p_gap_y[col] + C.gap_ext;
			gap_y[col] = max(open_y, extend_y);

			match[col] = max(d_score, max(gap_x[col], gap_y[col]));
		}

		for (s32 col = 0; col <= len2; col++) {
			p_match[col] = match[col];
			p_gap_y[col] = gap_y[col];
		}
	}

	s32 score = match[len2];
	if (!C.triangular) {
		scores[static_cast<s64>(C.num) * i + j] = score;
		scores[static_cast<s64>(C.num) * j + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(C.progress, 1);
}

extern "C" __global__ void kernel_sw(s32 *scores, s64 start, s64 batch)
{
	s64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	s64 alignment = start + tid;
	s32 j = d_find_j(alignment);
	s32 i = static_cast<s32>(alignment - alignments(static_cast<s64>(j)));

	s32 len1 = C.meta[i].len;
	s32 len2 = C.meta[j].len;

	s32 match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (s32 col = 0; col <= len2; col++) {
		match[col] = 0;
		gap_x[col] = gap_y[col] = SCORE_MIN;
	}

	s32 p_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 p_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (s32 col = 0; col <= len2; col++) {
		p_match[col] = match[col];
		p_gap_y[col] = gap_y[col];
	}

	s32 max_score = 0;
	for (s32 row = 1; row <= len1; ++row) {
		match[0] = 0;
		gap_x[0] = gap_y[0] = SCORE_MIN;

		s32 c1 = d_seq_lut(i, row - 1);
		for (s32 col = 1; col <= len2; col++) {
			s32 c2 = d_seq_lut(j, col - 1);
			s32 similarity = d_sub_mat(c1, c2);

			s32 d_score = p_match[col - 1] + similarity;

			s32 open_x = match[col - 1] + C.gap_open;
			s32 extend_x = gap_x[col - 1] + C.gap_ext;
			gap_x[col] = max(open_x, extend_x);

			s32 open_y = p_match[col] + C.gap_open;
			s32 extend_y = p_gap_y[col] + C.gap_ext;
			gap_y[col] = max(open_y, extend_y);

			s32 best = max(0, max(d_score,
					      max(gap_x[col], gap_y[col])));
			match[col] = best;
			if (best > max_score)
				max_score = best;
		}

		for (s32 col = 0; col <= len2; col++) {
			p_match[col] = match[col];
			p_gap_y[col] = gap_y[col];
		}
	}

	s32 score = max_score;
	if (!C.triangular) {
		scores[static_cast<s64>(C.num) * i + j] = score;
		scores[static_cast<s64>(C.num) * j + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(C.progress, 1);
}
