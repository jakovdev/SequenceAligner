#include "bio/algorithm/alignment.cuh"

__constant__ Constants C;

__forceinline__ __device__ s32 d_seq_lut(const s32 ij, const s32 pos)
{
	return C.seq_lut[(uchar)C.letters[C.offsets[ij] + pos]];
}

__forceinline__ __device__ s32 d_sub_mat(const s32 c1, const s32 c2)
{
	return C.sub_mat[c1 * SUB_MAT_DIM + c2];
}

__forceinline__ __device__ s32 d_find_j(const s64 alignment)
{
	s32 low = 1, high = C.seq_n;

	while (low < high) {
		const s32 mid = low + (high - low) / 2;
		if (((s64)mid * (mid - 1)) / 2 <= alignment)
			low = mid + 1;
		else
			high = mid;
	}

	return low - 1;
}

__global__ void k_nw(s32 *scores, s64 start, s64 batch)
{
	const s64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	const s64 alignment = start + tid;
	const s32 j = d_find_j(alignment);
	const s32 i = static_cast<s32>(alignment - ((s64)j * (j - 1)) / 2);

	const s32 len1 = C.lengths[i];
	const s32 len2 = C.lengths[j];

	s32 dp_prev[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 dp_curr[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (s32 col = 0; col <= len2; col++)
		dp_prev[col] = col * C.gap_pen;
	for (s32 row = 1; row <= len1; ++row) {
		dp_curr[0] = row * C.gap_pen;

		for (s32 col = 1; col <= len2; col++) {
			const s32 c1 = d_seq_lut(i, row - 1);
			const s32 c2 = d_seq_lut(j, col - 1);
			const s32 match = dp_prev[col - 1] + d_sub_mat(c1, c2);
			const s32 gap_v = dp_prev[col] + C.gap_pen;
			const s32 gap_h = dp_curr[col - 1] + C.gap_pen;

			s32 max = match > gap_v ? match : gap_v;
			max = max > gap_h ? max : gap_h;
			dp_curr[col] = max;
		}

		for (s32 col = 0; col <= len2; col++)
			dp_prev[col] = dp_curr[col];
	}

	const s32 score = dp_prev[len2];
	if (!C.triangular) {
		scores[static_cast<s64>(C.seq_n) * i + j] = score;
		scores[static_cast<s64>(C.seq_n) * j + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(reinterpret_cast<ull *>(C.checksum), static_cast<ull>(score));
	atomicAdd(reinterpret_cast<ull *>(C.progress), 1);
}

__global__ void k_ga(s32 *scores, s64 start, s64 batch)
{
	const s64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	const s64 alignment = start + tid;
	const s32 j = d_find_j(alignment);
	const s32 i = static_cast<s32>(alignment - ((s64)j * (j - 1)) / 2);

	const s32 len1 = C.lengths[i];
	const s32 len2 = C.lengths[j];

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

		const s32 c1 = d_seq_lut(i, row - 1);
		for (s32 col = 1; col <= len2; col++) {
			const s32 c2 = d_seq_lut(j, col - 1);
			const s32 similarity = d_sub_mat(c1, c2);

			const s32 d_score = p_match[col - 1] + similarity;

			const s32 open_x = match[col - 1] + C.gap_open;
			const s32 extend_x = gap_x[col - 1] + C.gap_ext;
			gap_x[col] = max(open_x, extend_x);

			const s32 open_y = p_match[col] + C.gap_open;
			const s32 extend_y = p_gap_y[col] + C.gap_ext;
			gap_y[col] = max(open_y, extend_y);

			match[col] = max(d_score, max(gap_x[col], gap_y[col]));
		}

		for (s32 col = 0; col <= len2; col++) {
			p_match[col] = match[col];
			p_gap_y[col] = gap_y[col];
		}
	}

	const s32 score = match[len2];
	if (!C.triangular) {
		scores[static_cast<s64>(C.seq_n) * i + j] = score;
		scores[static_cast<s64>(C.seq_n) * j + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(reinterpret_cast<ull *>(C.checksum), static_cast<ull>(score));
	atomicAdd(reinterpret_cast<ull *>(C.progress), 1);
}

__global__ void k_sw(s32 *scores, s64 start, s64 batch)
{
	const s64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	const s64 alignment = start + tid;
	const s32 j = d_find_j(alignment);
	const s32 i = static_cast<s32>(alignment - ((s64)j * (j - 1)) / 2);

	const s32 len1 = C.lengths[i];
	const s32 len2 = C.lengths[j];

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

		const s32 c1 = d_seq_lut(i, row - 1);
		for (s32 col = 1; col <= len2; col++) {
			const s32 c2 = d_seq_lut(j, col - 1);
			const s32 similarity = d_sub_mat(c1, c2);

			const s32 d_score = p_match[col - 1] + similarity;

			const s32 open_x = match[col - 1] + C.gap_open;
			const s32 extend_x = gap_x[col - 1] + C.gap_ext;
			gap_x[col] = max(open_x, extend_x);

			const s32 open_y = p_match[col] + C.gap_open;
			const s32 extend_y = p_gap_y[col] + C.gap_ext;
			gap_y[col] = max(open_y, extend_y);

			const s32 best = max(
				0, max(d_score, max(gap_x[col], gap_y[col])));
			match[col] = best;
			if (best > max_score)
				max_score = best;
		}

		for (s32 col = 0; col <= len2; col++) {
			p_match[col] = match[col];
			p_gap_y[col] = gap_y[col];
		}
	}

	const s32 score = max_score;
	if (!C.triangular) {
		scores[static_cast<s64>(C.seq_n) * i + j] = score;
		scores[static_cast<s64>(C.seq_n) * j + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(reinterpret_cast<ull *>(C.checksum), static_cast<ull>(score));
	atomicAdd(reinterpret_cast<ull *>(C.progress), 1);
}

extern "C" {
cudaError_t copy_constants(const struct Constants *host)
{
	return cudaMemcpyToSymbol(C, host, sizeof(C));
}

const void *kernel_function(enum AlignmentMethod method)
{
	static const void *k[ALIGN_COUNT] = {
		[ALIGN_GOTOH_AFFINE] = (const void *)k_ga,
		[ALIGN_NEEDLEMAN_WUNSCH] = (const void *)k_nw,
		[ALIGN_SMITH_WATERMAN] = (const void *)k_sw,
	};
	return k[method];
}
}
