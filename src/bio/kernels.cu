#include "bio/kernels.cuh"
#include "util/macros.h"

__constant__ struct constants C;

__forceinline__ __device__ shz d_seq_lut(uhz ij, uhz pos)
{
	return C.seq_lut[C.letters[C.meta[ij].off + pos]];
}

__forceinline__ __device__ shz d_sub_mat(shz c1, shz c2)
{
	return C.sub_mat[c1 * SUB_MAT_DIM + c2];
}

__forceinline__ __device__ uhz d_find_j(usz alignment)
{
	uhz low = 1, high = C.num;

	while (low < high) {
		uhz mid = low + (high - low) / 2;
		if (alignments(static_cast<usz>(mid)) <= alignment)
			low = mid + 1;
		else
			high = mid;
	}

	return low - 1;
}

extern "C" __global__ void kernel_nw(shz *scores, usz start, usz batch)
{
	usz tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	usz alignment = start + tid;
	uhz j = d_find_j(alignment);
	uhz i = static_cast<uhz>(alignment - alignments(static_cast<usz>(j)));

	uhz len1 = C.meta[i].len;
	uhz len2 = C.meta[j].len;

	shz dp_prev[MAX_CUDA_SEQUENCE_LENGTH + 1];
	shz dp_curr[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (uhz col = 0; col <= len2; col++)
		dp_prev[col] = col * C.gap_pen;
	for (uhz row = 1; row <= len1; ++row) {
		dp_curr[0] = row * C.gap_pen;

		for (uhz col = 1; col <= len2; col++) {
			shz c1 = d_seq_lut(i, row - 1);
			shz c2 = d_seq_lut(j, col - 1);
			shz match = dp_prev[col - 1] + d_sub_mat(c1, c2);
			shz gap_v = dp_prev[col] + C.gap_pen;
			shz gap_h = dp_curr[col - 1] + C.gap_pen;

			shz val_max = max(match, gap_v);
			val_max = max(val_max, gap_h);
			dp_curr[col] = val_max;
		}

		for (uhz col = 0; col <= len2; col++)
			dp_prev[col] = dp_curr[col];
	}

	shz score = dp_prev[len2];
	if (!C.triangular) {
		scores[static_cast<usz>(C.num) * i + j] = score;
		scores[static_cast<usz>(C.num) * j + i] = score;
	} else {
		scores[tid] = score;
	}
}

extern "C" __global__ void kernel_ga(shz *scores, usz start, usz batch)
{
	usz tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	usz alignment = start + tid;
	uhz j = d_find_j(alignment);
	uhz i = static_cast<uhz>(alignment - alignments(static_cast<usz>(j)));

	uhz len1 = C.meta[i].len;
	uhz len2 = C.meta[j].len;

	shz match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	shz gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
	shz gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;
	for (uhz col = 1; col <= len2; col++) {
		gap_x[col] = max(match[col - 1] + C.gap_opn,
				 gap_x[col - 1] + C.gap_ext);
		match[col] = gap_x[col];
		gap_y[col] = SCORE_MIN;
	}

	shz p_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	shz p_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (uhz col = 0; col <= len2; col++) {
		p_match[col] = match[col];
		p_gap_y[col] = gap_y[col];
	}

	for (uhz row = 1; row <= len1; ++row) {
		gap_x[0] = SCORE_MIN;
		gap_y[0] = max(p_match[0] + C.gap_opn, p_gap_y[0] + C.gap_ext);
		match[0] = gap_y[0];

		shz c1 = d_seq_lut(i, row - 1);
		for (uhz col = 1; col <= len2; col++) {
			shz c2 = d_seq_lut(j, col - 1);
			shz similarity = d_sub_mat(c1, c2);

			shz d_score = p_match[col - 1] + similarity;

			shz open_x = match[col - 1] + C.gap_opn;
			shz extend_x = gap_x[col - 1] + C.gap_ext;
			gap_x[col] = max(open_x, extend_x);

			shz open_y = p_match[col] + C.gap_opn;
			shz extend_y = p_gap_y[col] + C.gap_ext;
			gap_y[col] = max(open_y, extend_y);

			match[col] = max(d_score, max(gap_x[col], gap_y[col]));
		}

		for (uhz col = 0; col <= len2; col++) {
			p_match[col] = match[col];
			p_gap_y[col] = gap_y[col];
		}
	}

	shz score = match[len2];
	if (!C.triangular) {
		scores[static_cast<usz>(C.num) * i + j] = score;
		scores[static_cast<usz>(C.num) * j + i] = score;
	} else {
		scores[tid] = score;
	}
}

extern "C" __global__ void kernel_sw(shz *scores, usz start, usz batch)
{
	usz tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	usz alignment = start + tid;
	uhz j = d_find_j(alignment);
	uhz i = static_cast<uhz>(alignment - alignments(static_cast<usz>(j)));

	uhz len1 = C.meta[i].len;
	uhz len2 = C.meta[j].len;

	shz match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	shz gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
	shz gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (uhz col = 0; col <= len2; col++) {
		match[col] = 0;
		gap_x[col] = gap_y[col] = SCORE_MIN;
	}

	shz p_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	shz p_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (uhz col = 0; col <= len2; col++) {
		p_match[col] = match[col];
		p_gap_y[col] = gap_y[col];
	}

	shz max_score = 0;
	for (uhz row = 1; row <= len1; ++row) {
		match[0] = 0;
		gap_x[0] = gap_y[0] = SCORE_MIN;

		shz c1 = d_seq_lut(i, row - 1);
		for (uhz col = 1; col <= len2; col++) {
			shz c2 = d_seq_lut(j, col - 1);
			shz similarity = d_sub_mat(c1, c2);

			shz d_score = p_match[col - 1] + similarity;

			shz open_x = match[col - 1] + C.gap_opn;
			shz extend_x = gap_x[col - 1] + C.gap_ext;
			gap_x[col] = max(open_x, extend_x);

			shz open_y = p_match[col] + C.gap_opn;
			shz extend_y = p_gap_y[col] + C.gap_ext;
			gap_y[col] = max(open_y, extend_y);

			shz best = max(0, max(d_score,
					      max(gap_x[col], gap_y[col])));
			match[col] = best;
			if (best > max_score)
				max_score = best;
		}

		for (uhz col = 0; col <= len2; col++) {
			p_match[col] = match[col];
			p_gap_y[col] = gap_y[col];
		}
	}

	shz score = max_score;
	if (!C.triangular) {
		scores[static_cast<usz>(C.num) * i + j] = score;
		scores[static_cast<usz>(C.num) * j + i] = score;
	} else {
		scores[tid] = score;
	}
}
