#include "cuda_manager.cuh"
#include "host_types.h"

/*
Increase if needed, this depends on your available VRAM.
On an 8GB card, the limit is around 20000-22000.
If over the limit, it will write out of memory error message.
*/
#define MAX_CUDA_SEQUENCE_LENGTH (1024)

#define SCORE_MATDIM (24)
#define SEQ_LOOKUP_SIZ (128)
__constant__ int c_score_mat[SCORE_MATDIM * SCORE_MATDIM];
#define SCORE_MAT(i, j) c_score_mat[(i) * SCORE_MATDIM + (j)]
__constant__ int c_seq_lookup[SEQ_LOOKUP_SIZ];

__constant__ int c_gap_penalty;
__constant__ int c_gap_open;
__constant__ int c_gap_extend;
__constant__ bool c_triangular = false;

bool Cuda::copyTriangularMatrixFlag(bool triangular)
{
	if (!m_init) {
		setHostError("CUDA not initialized");
		return false;
	}

	cudaError_t err;
	m_kr.d_triangular = triangular;
	C_COPY(c_triangular, &triangular, sizeof(c_triangular));

	return true;
}

bool Cuda::uploadScoring(int *scoring_matrix, int *sequence_lookup)
{
	if (!m_init) {
		setHostError("CUDA not initialized");
		return false;
	}

	cudaError_t err;

	C_COPY(c_score_mat, scoring_matrix, sizeof(c_score_mat));
	C_COPY(c_seq_lookup, sequence_lookup, sizeof(c_seq_lookup));

	return true;
}

bool Cuda::uploadPenalties(s32 linear, s32 open, s32 extend)
{
	if (!m_init) {
		setHostError("CUDA not initialized");
		return false;
	}

	cudaError_t err;

	C_COPY(c_gap_penalty, &linear, sizeof(c_gap_penalty));
	C_COPY(c_gap_open, &open, sizeof(c_gap_open));
	C_COPY(c_gap_extend, &extend, sizeof(c_gap_extend));

	return true;
}

__forceinline__ __device__ uchar d_seq_letter(const Sequences *const seqs,
					      const u32 ij, const u32 pos)
{
	return (uchar)seqs->d_letters[seqs->d_offsets[ij] + pos];
}

__forceinline__ __device__ u32 d_seq_length(const Sequences *const seqs,
					    const u32 ij)
{
	return seqs->d_lengths[ij];
}

__forceinline__ __device__ u64 d_tri_idx(const Sequences *const seqs,
					 const u32 j)
{
	return seqs->d_indices[j];
}

template <typename T>
__forceinline__ __device__ u32 d_binary_search(const T *const elements,
					       const u32 length,
					       const u64 target)
{
	u32 low = 1, high = length - 1;
	u32 result = 1;

	while (low <= high) {
		u32 mid = (low + high) / 2;

		if (elements[mid] <= target) {
			if (mid + 1 >= length || elements[mid + 1] > target) {
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

__forceinline__ __device__ u32 find_sequence_column_binary_search(
	const Sequences *const seqs, const u64 alignment)
{
	return d_binary_search(seqs->d_indices, seqs->n_seqs, alignment);
}

__global__ void k_nw(const Sequences seqs, s32 *R scores, ull *R progress,
		     sll *R sum, u64 start, u64 batch)
{
	const u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	const u64 alignment = start + tid;
	const u32 n = seqs.n_seqs;

	const u32 j = find_sequence_column_binary_search(&seqs, alignment);
	const u32 i = alignment - d_tri_idx(&seqs, j);

	if (i >= n || j >= n || i >= j) {
		// Should never happen
		atomicAdd(progress, 1);
		return;
	}

	const u32 len1 = d_seq_length(&seqs, i);
	const u32 len2 = d_seq_length(&seqs, j);

	if (len1 > MAX_CUDA_SEQUENCE_LENGTH ||
	    len2 > MAX_CUDA_SEQUENCE_LENGTH) {
		// Temporary fix, ignore for now
		atomicAdd(progress, 1);
		return;
	}

	s32 dp_prev[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 dp_curr[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (u32 col = 0; col <= len2; col++)
		dp_prev[col] = col * -(c_gap_penalty);
	for (u32 row = 1; row <= len1; ++row) {
		dp_curr[0] = row * -(c_gap_penalty);

		for (u32 col = 1; col <= len2; col++) {
			const int idx1 =
				c_seq_lookup[d_seq_letter(&seqs, i, row - 1)];
			const int idx2 =
				c_seq_lookup[d_seq_letter(&seqs, j, col - 1)];
			const s32 match =
				dp_prev[col - 1] + SCORE_MAT(idx1, idx2);
			const s32 gap_v = dp_prev[col] - c_gap_penalty;
			const s32 gap_h = dp_curr[col - 1] - c_gap_penalty;

			s32 max = match > gap_v ? match : gap_v;
			max = max > gap_h ? max : gap_h;
			dp_curr[col] = max;
		}

		for (u32 col = 0; col <= len2; col++)
			dp_prev[col] = dp_curr[col];
	}

	const s32 score = dp_prev[len2];
	if (!c_triangular) {
		scores[(u64)i * n + j] = score;
		scores[(u64)j * n + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(progress, 1);
	atomicAdd(reinterpret_cast<ull *>(sum), static_cast<ull>(score));
}

__global__ void k_ga(const Sequences seqs, s32 *R scores, ull *R progress,
		     sll *R sum, u64 start, u64 batch)
{
	const u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	const u64 alignment = start + tid;
	const u32 n = seqs.n_seqs;

	const u32 j = find_sequence_column_binary_search(&seqs, alignment);
	const u32 i = alignment - d_tri_idx(&seqs, j);

	if (i >= n || j >= n || i >= j) {
		// Should never happen
		atomicAdd(progress, 1);
		return;
	}

	const u32 len1 = d_seq_length(&seqs, i);
	const u32 len2 = d_seq_length(&seqs, j);

	if (len1 > MAX_CUDA_SEQUENCE_LENGTH ||
	    len2 > MAX_CUDA_SEQUENCE_LENGTH) {
		// Temporary fix, ignore for now
		atomicAdd(progress, 1);
		return;
	}

	s32 match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	match[0] = 0;
	gap_x[0] = gap_y[0] = SCORE_MIN;
	for (u32 col = 1; col <= len2; col++) {
		gap_x[col] = max(match[col - 1] - c_gap_open,
				 gap_x[col - 1] - c_gap_extend);
		match[col] = gap_x[col];
		gap_y[col] = SCORE_MIN;
	}

	s32 prev_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 prev_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (u32 col = 0; col <= len2; col++) {
		prev_match[col] = match[col];
		prev_gap_y[col] = gap_y[col];
	}

	for (u32 row = 1; row <= len1; ++row) {
		match[0] = row * -(c_gap_penalty);
		gap_x[0] = SCORE_MIN;
		gap_y[0] = max(prev_match[0] - c_gap_open,
			       prev_gap_y[0] - c_gap_extend);
		match[0] = gap_y[0];

		const int idx1 = c_seq_lookup[d_seq_letter(&seqs, i, row - 1)];
		for (u32 col = 1; col <= len2; col++) {
			const int idx2 =
				c_seq_lookup[d_seq_letter(&seqs, j, col - 1)];
			const s32 similarity = SCORE_MAT(idx1, idx2);

			const s32 diag_score = prev_match[col - 1] + similarity;

			const s32 open_x = match[col - 1] - c_gap_open;
			const s32 extend_x = gap_x[col - 1] - c_gap_extend;
			gap_x[col] = max(open_x, extend_x);

			const s32 open_y = prev_match[col] - c_gap_open;
			const s32 extend_y = prev_gap_y[col] - c_gap_extend;
			gap_y[col] = max(open_y, extend_y);

			match[col] =
				max(diag_score, max(gap_x[col], gap_y[col]));
		}

		for (u32 col = 0; col <= len2; col++) {
			prev_match[col] = match[col];
			prev_gap_y[col] = gap_y[col];
		}
	}

	const s32 score = match[len2];
	if (!c_triangular) {
		scores[(u64)i * n + j] = score;
		scores[(u64)j * n + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(progress, 1);
	atomicAdd(reinterpret_cast<ull *>(sum), static_cast<ull>(score));
}

__global__ void k_sw(const Sequences seqs, s32 *R scores, ull *R progress,
		     sll *R sum, u64 start, u64 batch)
{
	const u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch)
		return;

	const u64 alignment = start + tid;
	const u32 n = seqs.n_seqs;

	const u32 j = find_sequence_column_binary_search(&seqs, alignment);
	const u32 i = alignment - d_tri_idx(&seqs, j);

	if (i >= n || j >= n || i >= j) {
		// Should never happen
		atomicAdd(progress, 1);
		return;
	}

	const u32 len1 = d_seq_length(&seqs, i);
	const u32 len2 = d_seq_length(&seqs, j);

	if (len1 > MAX_CUDA_SEQUENCE_LENGTH ||
	    len2 > MAX_CUDA_SEQUENCE_LENGTH) {
		// Temporary fix, ignore for now
		atomicAdd(progress, 1);
		return;
	}

	s32 match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_x[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (u32 col = 0; col <= len2; col++) {
		match[col] = 0;
		gap_x[col] = gap_y[col] = SCORE_MIN;
	}

	s32 prev_match[MAX_CUDA_SEQUENCE_LENGTH + 1];
	s32 prev_gap_y[MAX_CUDA_SEQUENCE_LENGTH + 1];
	for (u32 col = 0; col <= len2; col++) {
		prev_match[col] = match[col];
		prev_gap_y[col] = gap_y[col];
	}

	s32 max_score = 0;
	for (u32 row = 1; row <= len1; ++row) {
		match[0] = 0;
		gap_x[0] = gap_y[0] = SCORE_MIN;

		const int idx1 = c_seq_lookup[d_seq_letter(&seqs, i, row - 1)];
		for (u32 col = 1; col <= len2; col++) {
			const int idx2 =
				c_seq_lookup[d_seq_letter(&seqs, j, col - 1)];
			const s32 similarity = SCORE_MAT(idx1, idx2);

			const s32 diag_score = prev_match[col - 1] + similarity;

			const s32 open_x = match[col - 1] - c_gap_open;
			const s32 extend_x = gap_x[col - 1] - c_gap_extend;
			gap_x[col] = max(open_x, extend_x);

			const s32 open_y = prev_match[col] - c_gap_open;
			const s32 extend_y = prev_gap_y[col] - c_gap_extend;
			gap_y[col] = max(open_y, extend_y);

			const s32 best =
				max(0, max(diag_score,
					   max(gap_x[col], gap_y[col])));
			match[col] = best;
			if (best > max_score)
				max_score = best;
		}

		for (u32 col = 0; col <= len2; col++) {
			prev_match[col] = match[col];
			prev_gap_y[col] = gap_y[col];
		}
	}

	const s32 score = max_score;
	if (!c_triangular) {
		scores[(u64)i * n + j] = score;
		scores[(u64)j * n + i] = score;
	} else {
		scores[tid] = score;
	}

	atomicAdd(progress, 1);
	atomicAdd(reinterpret_cast<ull *>(sum), static_cast<ull>(score));
}

bool Cuda::switchKernel(int kernel_id)
{
	cudaError_t err;
	u64 offset = m_kr.h_last_batch;
	u64 batch = m_kr.h_batch_size;
	s32 *d_buffer = m_kr.d_scores0;

	if (offset >= m_kr.h_total_count) {
		if (m_kr.h_after_first) {
			err = cudaDeviceSynchronize();
			CUDA_ERROR(
				"Device synchronization failed on final check");
			DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);
			m_kr.h_active = 1 - m_kr.h_active;
		}

		return true;
	}

	if (m_kr.d_triangular) {
		offset = m_kr.h_last_batch;
		if (offset + batch > m_kr.h_total_count)
			batch = m_kr.h_total_count - offset;

		if (!batch) {
			if (m_kr.h_after_first) {
				err = cudaDeviceSynchronize();
				CUDA_ERROR(
					"Device synchronization failed on final check");
				DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);
			}

			return true;
		}

		if (m_kr.h_after_first) {
			err = cudaDeviceSynchronize();
			CUDA_ERROR("Device synchronization failed");
			DH_COPY(&m_kr.h_progress, m_kr.d_progress, 1);
			m_kr.h_active = 1 - m_kr.h_active;
		}

		if (m_kr.h_active == 1)
			d_buffer = m_kr.d_scores1;
	}

	const uint block_dim = (uint)(m_dev.maxThreadsPerBlock);
	const uint grid_dim = (uint)((batch + block_dim - 1) / block_dim);

	if (!grid_dim || grid_dim > (uint)(m_dev.maxGridSize[0])) {
		setHostError("Grid size exceeds device limit");
		return false;
	}

	ull *d_progress = m_kr.d_progress;
	sll *d_checksum = m_kr.d_checksum;

	switch (kernel_id) {
	default:
		setHostError("Invalid kernel ID");
		return false;
	case 0: // Gotoh Affine
		k_ga<<<grid_dim, block_dim, 0, m_kr.s_comp>>>(m_seqs, d_buffer,
							      d_progress,
							      d_checksum,
							      offset, batch);
		break;
	case 1: // Needleman-Wunsch
		k_nw<<<grid_dim, block_dim, 0, m_kr.s_comp>>>(m_seqs, d_buffer,
							      d_progress,
							      d_checksum,
							      offset, batch);
		break;
	case 2: // Smith-Waterman
		k_sw<<<grid_dim, block_dim, 0, m_kr.s_comp>>>(m_seqs, d_buffer,
							      d_progress,
							      d_checksum,
							      offset, batch);
		break;
	}

	err = cudaGetLastError();
	CUDA_ERROR("Kernel launch failed");

	m_kr.h_last_batch += batch;

	return true;
}
