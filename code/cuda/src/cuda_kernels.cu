#include "cuda_manager.cuh"

#include <cstring>

__constant__ Constants C;

__forceinline__ __device__ s32 d_seq_lup(const u32 ij, const u32 pos)
{
	return C.seq_lup[(uchar)C.letters[C.offsets[ij] + pos]];
}

__forceinline__ __device__ s32 d_sub_mat(const s32 a, const s32 b)
{
	return C.sub_mat[a * SUB_MATDIM + b];
}

__forceinline__ __device__ u32 d_find_j(const u64 id)
{
	u32 low = 1, high = C.seqs_n - 1;
	u32 result = 1;

	while (low <= high) {
		const u32 mid = (low + high) / 2;

		if (C.indices[mid] <= id) {
			if (mid + 1 >= C.seqs_n || C.indices[mid + 1] > id) {
				result = mid;
				break;
			}

			low = mid + 1;
		} else {
			high = mid - 1;
		}
	}

	return result;
}

__global__ void k_nw(s32 *R scores, u64 start, u64 batch)
{
	const u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	const u64 alignment = start + tid;
	const u32 j = d_find_j(alignment);
	const u32 i = alignment - C.indices[j];

	const u32 len1 = C.lengths[i];
	const u32 len2 = C.lengths[j];

	if (len1 > MAX_CUDA_SEQUENCE_LENGTH ||
	    len2 > MAX_CUDA_SEQUENCE_LENGTH) {
		// Temporary fix, ignore for now
		atomicAdd(C.progress, 1);
		return;
	}

	s32 dp_prev[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 dp_curr[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (u32 col = 0; col <= len2; col++)
		dp_prev[col] = col * -(C.gap_pen);
	for (u32 row = 1; row <= len1; ++row) {
		dp_curr[0] = row * -(C.gap_pen);

		for (u32 col = 1; col <= len2; col++) {
			const s32 c1 = d_seq_lup(i, row - 1);
			const s32 c2 = d_seq_lup(j, col - 1);
			const s32 match = dp_prev[col - 1] + d_sub_mat(c1, c2);
			const s32 gap_v = dp_prev[col] - C.gap_pen;
			const s32 gap_h = dp_curr[col - 1] - C.gap_pen;

			s32 max = match > gap_v ? match : gap_v;
			max = max > gap_h ? max : gap_h;
			dp_curr[col] = max;
		}

		for (u32 col = 0; col <= len2; col++)
			dp_prev[col] = dp_curr[col];
	}

	const s32 score = dp_prev[len2];
	if (!C.triangular) {
		scores[(u64)i * C.seqs_n + j] = score;
		scores[(u64)j * C.seqs_n + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(reinterpret_cast<ull *>(C.checksum), static_cast<ull>(score));
	atomicAdd(C.progress, 1);
}

__global__ void k_ga(s32 *R scores, u64 start, u64 batch)
{
	const u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	const u64 alignment = start + tid;
	const u32 j = d_find_j(alignment);
	const u32 i = alignment - C.indices[j];

	const u32 len1 = C.lengths[i];
	const u32 len2 = C.lengths[j];

	if (len1 > MAX_CUDA_SEQUENCE_LENGTH ||
	    len2 > MAX_CUDA_SEQUENCE_LENGTH) {
		// Temporary fix, ignore for now
		atomicAdd(C.progress, 1);
		return;
	}

	s32 match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;
	for (u32 col = 1; col <= len2; col++) {
		gap_x[col] = max(match[col - 1] - C.gap_open,
				 gap_x[col - 1] - C.gap_ext);
		match[col] = gap_x[col];
		gap_y[col] = SCORE_MIN;
	}

	s32 p_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 p_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (u32 col = 0; col <= len2; col++) {
		p_match[col] = match[col];
		p_gap_y[col] = gap_y[col];
	}

	for (u32 row = 1; row <= len1; ++row) {
		match[0] = row * -(C.gap_pen);
		gap_x[0] = SCORE_MIN;
		gap_y[0] = max(p_match[0] - C.gap_open, p_gap_y[0] - C.gap_ext);
		match[0] = gap_y[0];

		const s32 c1 = d_seq_lup(i, row - 1);
		for (u32 col = 1; col <= len2; col++) {
			const s32 c2 = d_seq_lup(j, col - 1);
			const s32 similarity = d_sub_mat(c1, c2);

			const s32 d_score = p_match[col - 1] + similarity;

			const s32 open_x = match[col - 1] - C.gap_open;
			const s32 extend_x = gap_x[col - 1] - C.gap_ext;
			gap_x[col] = max(open_x, extend_x);

			const s32 open_y = p_match[col] - C.gap_open;
			const s32 extend_y = p_gap_y[col] - C.gap_ext;
			gap_y[col] = max(open_y, extend_y);

			match[col] = max(d_score, max(gap_x[col], gap_y[col]));
		}

		for (u32 col = 0; col <= len2; col++) {
			p_match[col] = match[col];
			p_gap_y[col] = gap_y[col];
		}
	}

	const s32 score = match[len2];
	if (!C.triangular) {
		scores[(u64)i * C.seqs_n + j] = score;
		scores[(u64)j * C.seqs_n + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(C.progress, 1);
	atomicAdd(reinterpret_cast<ull *>(C.checksum), static_cast<ull>(score));
}

__global__ void k_sw(s32 *R scores, u64 start, u64 batch)
{
	const u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	const u64 alignment = start + tid;
	const u32 j = d_find_j(alignment);
	const u32 i = alignment - C.indices[j];

	const u32 len1 = C.lengths[i];
	const u32 len2 = C.lengths[j];

	if (len1 > MAX_CUDA_SEQUENCE_LENGTH ||
	    len2 > MAX_CUDA_SEQUENCE_LENGTH) {
		// Temporary fix, ignore for now
		atomicAdd(C.progress, 1);
		return;
	}

	s32 match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (u32 col = 0; col <= len2; col++) {
		match[col] = 0;
		gap_x[col] = gap_y[col] = SCORE_MIN;
	}

	s32 p_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 p_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (u32 col = 0; col <= len2; col++) {
		p_match[col] = match[col];
		p_gap_y[col] = gap_y[col];
	}

	s32 max_score = 0;
	for (u32 row = 1; row <= len1; ++row) {
		match[0] = 0;
		gap_x[0] = gap_y[0] = SCORE_MIN;

		const s32 c1 = d_seq_lup(i, row - 1);
		for (u32 col = 1; col <= len2; col++) {
			const s32 c2 = d_seq_lup(j, col - 1);
			const s32 similarity = d_sub_mat(c1, c2);

			const s32 d_score = p_match[col - 1] + similarity;

			const s32 open_x = match[col - 1] - C.gap_open;
			const s32 extend_x = gap_x[col - 1] - C.gap_ext;
			gap_x[col] = max(open_x, extend_x);

			const s32 open_y = p_match[col] - C.gap_open;
			const s32 extend_y = p_gap_y[col] - C.gap_ext;
			gap_y[col] = max(open_y, extend_y);

			const s32 best = max(
				0, max(d_score, max(gap_x[col], gap_y[col])));
			match[col] = best;
			if (best > max_score)
				max_score = best;
		}

		for (u32 col = 0; col <= len2; col++) {
			p_match[col] = match[col];
			p_gap_y[col] = gap_y[col];
		}
	}

	const s32 score = max_score;
	if (!C.triangular) {
		scores[(u64)i * C.seqs_n + j] = score;
		scores[(u64)j * C.seqs_n + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(C.progress, 1);
	atomicAdd(reinterpret_cast<ull *>(C.checksum), static_cast<ull>(score));
}

bool Cuda::kernelLaunch(int kernel_id) noexcept
{
	if (!s.ready() || kernel_id > 2 || kernel_id < 0) {
		hostError("Invalid kernel ID or upload steps not completed");
		return false;
	}

	cudaError_t err;

	if (!h.subsequent) {
		D_MALLOC(d.progress, 1);
		D_MALLOC(d.checksum, 1);

		D_MEMSET(d.progress, 0, 1);
		D_MEMSET(d.checksum, 0, 1);

		C_COPY(C, static_cast<Constants *>(&d));
	}

	u64 offset = h.batch_last;
	u64 batch = h.batch;

	if (offset >= h.alignments) {
		if (h.subsequent) {
			D_SYNC("Device synchronization failed on final check");
			DH_COPY(&h.progress, d.progress, 1);
			d.swap();
		}

		return true;
	}

	if (d.triangular) {
		if (offset + batch > h.alignments)
			batch = h.alignments - offset;

		if (!batch) {
			if (h.subsequent) {
				D_SYNC("Device synchronization failed on final check");
				DH_COPY(&h.progress, d.progress, 1);
			}

			return true;
		}

		if (h.subsequent) {
			D_SYNC("Device synchronization failed");
			DH_COPY(&h.progress, d.progress, 1);
			d.swap();
		}
	}

	const uint gdim = (uint)((batch + d.bdim - 1) / d.bdim);
	if (!gdim || gdim > d.gdim_max) {
		hostError("Grid size exceeds device limit");
		return false;
	}

	switch (kernel_id) {
	default:
		hostError("Invalid kernel ID");
		return false;
	case 0: /* Gotoh Affine */
		k_ga<<<gdim, d.bdim, 0, d.s_comp>>>(d.current(), offset, batch);
		break;
	case 1: /* Needleman-Wunsch */
		k_nw<<<gdim, d.bdim, 0, d.s_comp>>>(d.current(), offset, batch);
		break;
	case 2: /* Smith-Waterman */
		k_sw<<<gdim, d.bdim, 0, d.s_comp>>>(d.current(), offset, batch);
		break;
	}

	err = cudaGetLastError();
	CUDA_ERROR("Kernel launch failed", false);

	h.batch_last += batch;

	return true;
}

bool Cuda::kernelResults() noexcept
{
	if (!s.ready()) {
		hostError("Invalid context or upload steps not completed");
		return false;
	}

	cudaError_t err;

	if (!d.triangular) {
		static bool matrix_copied = false;
		if (matrix_copied)
			return true;

		S_SYNC(d.s_comp);
		DH_COPY(&h.progress, d.progress, 1);
		DH_COPY(h.scores, d.scores[0], d.seqs_n * d.seqs_n);
		matrix_copied = true;
		return true;
	}

	if (h.batch_done >= h.alignments) {
		if (d.s_sync) {
			S_SYNC(d.s_copy);
			d.s_sync = false;
		}

		return true;
	}

	if (d.s_sync) {
		S_QUERY(d.s_copy);
		d.s_sync = false;
	}

	if (!h.subsequent && h.batch < h.alignments) {
		h.subsequent = true;
		return true;
	}

	u64 n_scores = std::min(h.batch, h.alignments - h.batch_done);
	if (!n_scores)
		return true;

	if (h.subsequent) {
		DH_ACOPY(h.scores + h.batch_done, d.next(), n_scores, d.s_copy);
		d.s_sync = true;
	} else {
		S_SYNC(d.s_comp);
		DH_COPY(&h.progress, d.progress, 1);
		DH_COPY(h.scores + h.batch_done, d.current(), n_scores);
	}

	h.batch_done += n_scores;
	return true;
}

ull Cuda::kernelProgress() const noexcept
{
	if (!s.ready() || !d.progress)
		return 0;

	return h.progress;
}

sll Cuda::kernelChecksum() noexcept
{
	if (!s.ready() || !d.checksum)
		return 0;

	cudaError_t err;
	S_SYNC(d.s_comp);

	sll checksum = 0;
	DH_COPY(&checksum, d.checksum, 1);
	return checksum;
}
