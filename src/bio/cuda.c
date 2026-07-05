#include <args.h>
#include <print.h>

#include "system/os.h"

#ifdef USE_CUDA
#ifdef __MINGW64__
#undef __cdecl
#endif
#include <cuda_runtime_api.h>
#include <string.h>
#include <stdlib.h>

#include "bio/align.h"
#include "bio/kernels.cuh"
#include "io/input.h"
#include "io/output.h"
#include "util/benchmark.h"
#include "util/macros.h"

#define CALL(cuda_func)                                            \
	do {                                                       \
		err = cuda_func;                                   \
		if (err != cudaSuccess) {                          \
			perr("CUDA: %s", cudaGetErrorString(err)); \
			goto ask_cuda;                             \
		}                                                  \
	} while (0)

static bool no_cuda;

size_t memory_gpu(void)
{
	if (no_cuda)
		return 0;
	cudaError_t err;
	size_t free = 0;
	size_t total = 0;
	CALL(cudaMemGetInfo(&free, &total));
	return free;
ask_cuda:
	no_cuda = print_Yn("Would you like to switch to non-CUDA (CPU)?");
	if (!no_cuda)
		exit(EXIT_FAILURE);
	return 0;
}

bool align_cuda(struct input in, struct output out)
{
	if (no_cuda)
		return align_cpu(in, out);

	if (in.max > MAX_CUDA_SEQUENCE_LENGTH) {
		perr("Sequence length exceeds CUDA Device limits");
		goto ask_cuda;
	}

	cudaError_t err;
	unsigned int block_max = ({
		int device;
		CALL(cudaGetDevice(&device));
		struct cudaDeviceProp prop;
		CALL(cudaGetDeviceProperties(&prop, device));
		pinfo("Using CUDA device: %s", prop.name);
		prop.maxThreadsPerBlock;
	});

	struct constants C = {
		.num = in.num,
		.gap_pen = GAP_PEN,
		.gap_open = GAP_OPN,
		.gap_ext = GAP_EXT,
	};

	memcpy(C.seq_lut, SEQ_LUT, sizeof(SEQ_LUT));
	memcpy(C.sub_mat, SUB_MAT, sizeof(SUB_MAT));

	s32 num = in.num;
	s32 sum = in.meta[num - 1].off + in.meta[num - 1].len + 1;
	size_t meta_bytes = bytesof(in.meta, num);

	CALL(cudaMalloc((void **)&C.letters, sum));
	CALL(cudaMalloc((void **)&C.meta, meta_bytes));
	CALL(cudaMemcpy(C.letters, in.seqs, sum, cudaMemcpyHostToDevice));
	CALL(cudaMemcpy(C.meta, in.meta, meta_bytes, cudaMemcpyHostToDevice));

	s32 *matrix = out.matrix;
	s64 alignments = alignments((s64)num);
	constexpr s64 batch_size = 64 << 20;
	size_t free = 0;
	size_t total = 0;
	CALL(cudaMemGetInfo(&free, &total));
	pverb("GPU: %.2f GiB free / %.2f GiB total", (double)free / (double)GiB,
	      (double)total / (double)GiB);
	free = free * 3 / 4;
	if (bytesof(matrix, num * num) > free) {
		if (bytesof(matrix, alignments) > free) {
			if (bytesof(matrix, batch_size) > free) {
				perr("Not enough CUDA Device memory for alignment");
				goto ask_cuda;
			}
		}
		C.triangular = true;
	}

	if (out.triangular)
		C.triangular = true;

	s64 batch = 0, batch_last = 0, batch_done = 0;
	void *scores[2] = {};
	s32 active = 0;
	if (C.triangular) {
		batch = min(alignments, batch_size);
		CALL(cudaMalloc(&scores[0], bytesof(matrix, batch)));
		CALL(cudaMemset(scores[0], 0, bytesof(matrix, batch)));
		CALL(cudaMalloc(&scores[1], bytesof(matrix, batch)));
		CALL(cudaMemset(scores[1], 0, bytesof(matrix, batch)));
	} else {
		batch = alignments;
		CALL(cudaMalloc(&*scores, bytesof(matrix, num * num)));
		CALL(cudaMemset(*scores, 0, bytesof(matrix, num * num)));
	}

	CALL(cudaMalloc((void **)&C.progress, sizeof(*C.progress)));
	CALL(cudaMemset(C.progress, 0, sizeof(*C.progress)));
	CALL(cudaMemcpyToSymbol(pC, &C, sizeof(C), 0, cudaMemcpyHostToDevice));

	const void *kernel = ALIGN->kernel;
	dim3 block = { block_max, 1, 1 };
	cudaStream_t compute, memory;
	CALL(cudaStreamCreate(&compute));
	CALL(cudaStreamCreate(&memory));

	bool subsequent = false, syncing = false, matrix_copied = false;
	s64 progress = 0;

	pinfo("Performing %zu pairwise alignments", (size_t)alignments);

	ppercent(0, "Aligning sequences");
	bench_align_start();
	while (true) {
		s64 offset = batch_last;
		if (offset >= alignments) {
			if (subsequent) {
				CALL(cudaDeviceSynchronize());
				CALL(cudaMemcpy(&progress, C.progress,
						sizeof(progress),
						cudaMemcpyDeviceToHost));
				active = 1 - active;
			}
			goto cuda_results;
		}

		if (C.triangular) {
			if (offset + batch > alignments)
				batch = alignments - offset;
			if (!batch) {
				if (subsequent) {
					CALL(cudaDeviceSynchronize());
					CALL(cudaMemcpy(&progress, C.progress,
							sizeof(progress),
							cudaMemcpyDeviceToHost));
				}
				goto cuda_results;
			}
			if (subsequent) {
				CALL(cudaDeviceSynchronize());
				CALL(cudaMemcpy(&progress, C.progress,
						sizeof(progress),
						cudaMemcpyDeviceToHost));
				active = 1 - active;
			}
		}

		dim3 grid = { (batch + block.x - 1) / block.x, 1, 1 };
		void *args[] = { &scores[active], &offset, &batch };
		CALL(cudaLaunchKernel(kernel, grid, block, args, 0, compute));
		batch_last += batch;
cuda_results:

		if (!C.triangular) {
			if (matrix_copied)
				goto cuda_progress;

			CALL(cudaStreamSynchronize(compute));
			CALL(cudaMemcpy(&progress, C.progress, sizeof(progress),
					cudaMemcpyDeviceToHost));

			if (matrix)
				CALL(cudaMemcpy(matrix, *scores,
						bytesof(matrix, num * num),
						cudaMemcpyDeviceToHost));

			matrix_copied = true;
			goto cuda_progress;
		}

		if (batch_done >= alignments) {
			if (syncing) {
				CALL(cudaStreamSynchronize(memory));
				syncing = false;
			}
			goto cuda_progress;
		}

		if (syncing) {
			err = cudaStreamQuery(memory);
			if (err == cudaErrorNotReady)
				goto cuda_progress;
			CALL(err);
			syncing = false;
		}

		if (!subsequent && batch < alignments) {
			subsequent = true;
			goto cuda_progress;
		}

		s64 n_scores = min(batch, alignments - batch_done);
		if (!n_scores)
			goto cuda_progress;

		if (subsequent) {
			if (matrix)
				CALL(cudaMemcpyAsync(
					matrix + batch_done, scores[1 - active],
					bytesof(matrix, n_scores),
					cudaMemcpyDeviceToHost, memory));
			syncing = true;
		} else {
			CALL(cudaStreamSynchronize(compute));
			CALL(cudaMemcpy(&progress, C.progress, sizeof(progress),
					cudaMemcpyDeviceToHost));
			if (matrix)
				CALL(cudaMemcpy(matrix + batch_done,
						scores[active],
						bytesof(matrix, n_scores),
						cudaMemcpyDeviceToHost));
		}
		batch_done += n_scores;
cuda_progress:
		pproportc(progress / alignments, "Aligning sequences");
		if (progress >= alignments)
			break;
	}

	bench_align_end();
	ppercent(100, "Aligning sequences");
	bench_align_print();
	cudaDeviceReset();
	return true;
ask_cuda:
	no_cuda = print_Yn("Would you like to switch to non-CUDA (CPU)?");
	if (no_cuda)
		return align_cpu(in, out);
	return false;
}

static struct arg_callback validate_cuda(void)
{
	int device = 0;
	int count = 0;
	cudaError_t err = cudaGetDeviceCount(&count);
	if (!count || err != cudaSuccess) {
		pwarn("No CUDA Devices found, defaulting to CPU-only");
		goto disable_cuda;
	}

	if (count > 1) {
		const char **MALLOCA(names, count);
		struct cudaDeviceProp *MALLOCA(props, count);
		if (!props || !names)
			return ARG_INVALID("Out of memory, try -C, --no-cuda");
		for (int i = 0; i < count; i++) {
			CALL(cudaGetDeviceProperties(&props[i], i));
			names[i] = props[i].name;
		}
		pinfo("You have %d CUDA Devices available", count);
		device = pchoice(names, count, "Choose your CUDA Device");
		free(props);
		free(names);
	}
	CALL(cudaSetDevice(device));
	return ARG_VALID();
ask_cuda:
	if (!print_Yn("Would you like to switch to non-CUDA (CPU)?"))
		return ARG_INVALID("Try using -C, --no-cuda");
disable_cuda:
	no_cuda = true;
	return ARG_VALID();
}

static void print_cuda_enabled(void)

{
	pinfom("CUDA: Enabled");
}

ARG_EXTERN(compression);
ARG_EXTERN(threads);

ARGUMENT(disable_cuda) = {
	.opt = 'C',
	.lopt = "no-cuda",
	.help = "Disable CUDA",
	.set = &no_cuda,
	.validate_callback = validate_cuda,
	.validate_phase = ARG_CALLBACK_IF_UNSET,
	.action_callback = print_cuda_enabled,
	.action_phase = ARG_CALLBACK_IF_UNSET,
	.action_order = ARG_ORDER_AFTER(ARG(compression)),
	.help_order = ARG_ORDER_AFTER(ARG(threads)),
};

#else

size_t memory_gpu(void)
{
	return 0;
}

static void print_cuda_ignored(void)
{
	pwarnm("CUDA: Ignored");
}

ARG_EXTERN(compression);

ARGUMENT(disable_cuda) = {
	.opt = 'C',
	.lopt = "no-cuda",
	.arg_req = ARG_HIDDEN,
	.action_callback = print_cuda_ignored,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(compression)),
};

#endif /* USE_CUDA */
