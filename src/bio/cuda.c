#include <args.h>
#include <print.h>
#include <string.h>
#include <stdlib.h>

#include "bio/align.h"
#include "bio/kernels.cuh"
#include "io/input.h"
#include "io/output.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

#ifdef _WIN32
#define CuDLL "nvcuda.dll"
#define CuPFN(cuFunction) __stdcall int(*cuFunction)
#else
#define CuDLL "libcuda.so.1"
#define CuPFN(cuFunction) int(*cuFunction)
#endif

typedef uintptr_t dev;

#define cuCtxCreate cuCtxCreate_v2
#define cuModuleGetGlobal cuModuleGetGlobal_v2
#define cuMemGetInfo cuMemGetInfo_v2
#define cuMemAlloc cuMemAlloc_v2
#define cuMemcpyHtoD cuMemcpyHtoD_v2
#define cuMemcpyDtoH cuMemcpyDtoH_v2
#define cuMemcpyDtoHAsync cuMemcpyDtoHAsync_v2
#define cuMemsetD32 cuMemsetD32_v2
#define cuMemFree cuMemFree_v2
#define cuStreamDestroy cuStreamDestroy_v2
#define cuCtxDestroy cuCtxDestroy_v2

CuPFN(cuGetErrorString)(int, const char **);
CuPFN(cuInit)(unsigned int);
CuPFN(cuDeviceGetCount)(int *);
CuPFN(cuDeviceGet)(int *, int);
CuPFN(cuCtxCreate)(void **, unsigned int, int);
CuPFN(cuModuleLoadData)(void **, const void *);
CuPFN(cuModuleGetFunction)(void **, void *, const char *);
CuPFN(cuModuleGetGlobal)(dev *, usz *, void *, const char *);
CuPFN(cuDeviceGetAttribute)(int *, int, int);
CuPFN(cuStreamCreate)(void **, unsigned int);
CuPFN(cuMemGetInfo)(usz *, usz *);
CuPFN(cuMemAlloc)(dev *, usz);
CuPFN(cuMemcpyHtoD)(dev, const void *, usz);
CuPFN(cuMemcpyDtoH)(void *, dev, usz);
CuPFN(cuMemcpyDtoHAsync)(void *, dev, usz, void *);
CuPFN(cuMemsetD32)(dev, unsigned int, usz);
CuPFN(cuLaunchKernel)(void *, unsigned int, unsigned int, unsigned int,
		      unsigned int, unsigned int, unsigned int, unsigned int,
		      void *, void **, void **);
CuPFN(cuCtxSynchronize)(void);
CuPFN(cuStreamSynchronize)(void *);
CuPFN(cuStreamQuery)(void *);
CuPFN(cuMemFree)(dev);
CuPFN(cuStreamDestroy)(void *);
CuPFN(cuModuleUnload)(void *);
CuPFN(cuCtxDestroy)(void *);

#define STR(FN) #FN
#define CALL(function)                               \
	do {                                         \
		int err = (function);                \
		if (err) {                           \
			const char *msg;             \
			cuGetErrorString(err, &msg); \
			perr("CUDA: %s", msg);       \
			goto ask_cuda;               \
		}                                    \
	} while (0)

static struct gpu_nvidia {
	void *dll;
	void *ctx;
	void *module;
	void *kernel;
	dev constants;
	void *compute;
	void *memory;
	int bx;
	int device;
} cu;

bool no_cuda;

static void free_cuda(void)
{
	cuStreamDestroy(cu.compute);
	cuStreamDestroy(cu.memory);
	cuModuleUnload(cu.module);
	cuCtxDestroy(cu.ctx);
	dll_close(cu.dll);
	memset(&cu, 0, sizeof(cu));
}

static struct arg_callback init_cuda(void)
{
	if (!ALIGN)
		return ARG_VALID();

	cu.dll = dll_open(CuDLL);
	if (!cu.dll)
		goto disable_cuda;

#define SYM(FN)                                \
	do {                                   \
		FN = dll_sym(cu.dll, STR(FN)); \
		if (!FN) {                     \
			dll_close(cu.dll);     \
			cu.dll = nullptr;      \
			goto disable_cuda;     \
		}                              \
	} while (0)

	SYM(cuGetErrorString);
	SYM(cuInit);
	SYM(cuDeviceGetCount);
	SYM(cuDeviceGet);
	SYM(cuCtxCreate);
	SYM(cuModuleLoadData);
	SYM(cuModuleGetFunction);
	SYM(cuModuleGetGlobal);
	SYM(cuDeviceGetAttribute);
	SYM(cuStreamCreate);
	SYM(cuMemGetInfo);
	SYM(cuMemAlloc);
	SYM(cuMemcpyHtoD);
	SYM(cuMemcpyDtoH);
	SYM(cuMemcpyDtoHAsync);
	SYM(cuMemsetD32);
	SYM(cuLaunchKernel);
	SYM(cuCtxSynchronize);
	SYM(cuStreamSynchronize);
	SYM(cuStreamQuery);
	SYM(cuMemFree);
	SYM(cuStreamDestroy);
	SYM(cuModuleUnload);
	SYM(cuCtxDestroy);

	int count = 0;
	if (cuInit(0) || cuDeviceGetCount(&count) || !count) {
		pwarn("No CUDA Devices found, defaulting to CPU-only");
		goto disable_cuda;
	}

	int device = 0; /* TODO: Allow Multi-Device Execution */
	CALL(cuDeviceGet(&cu.device, device));
	CALL(cuCtxCreate(&cu.ctx, 0, cu.device));
	static const unsigned char kernels[] = {
#embed "../generated/kernels.fatbin"
	};
	CALL(cuModuleLoadData(&cu.module, kernels));
	CALL(cuModuleGetFunction(&cu.kernel, cu.module, ALIGN->kernel));
	usz constants_size = 0;
	CALL(cuModuleGetGlobal(&cu.constants, &constants_size, cu.module, "C"));
	if (constants_size != sizeof(struct constants)) {
		perr("CUDA Kernel got corrupted, please report this");
		goto ask_cuda;
	}
	CALL(cuDeviceGetAttribute(&cu.bx, 1 /*THREADS_PER_BLOCK*/, cu.device));
	CALL(cuStreamCreate(&cu.compute, 0 /*STREAM_DEFAULT*/));
	CALL(cuStreamCreate(&cu.memory, 0 /*STREAM_DEFAULT*/));
	atexit(free_cuda);
	return ARG_VALID();
ask_cuda:
	if (!print_Yn("Would you like to switch to non-CUDA (CPU)?"))
		return ARG_INVALID("You can also pass --no-cuda");
disable_cuda:
	no_cuda = true;
	return ARG_VALID();
}

usz memory_gpu(void)
{
	if (no_cuda)
		return 0;
	usz free = 0, total = 0;
	CALL(cuMemGetInfo(&free, &total));
	return free;
ask_cuda:
	no_cuda = print_Yn("Would you like to switch to non-CUDA (CPU)?");
	if (!no_cuda)
		exit(EXIT_FAILURE);
	return 0;
}

bool align_cuda(struct input in, struct output out)
{
	if (in.max > MAX_CUDA_SEQUENCE_LENGTH) {
		perr("Sequence length exceeds CUDA Device limits");
		goto ask_cuda;
	}

	struct constants C = {
		.num = in.num,
		.gap_pen = GAP_PEN,
		.gap_opn = GAP_OPN,
		.gap_ext = GAP_EXT,
	};

	memcpy(C.seq_lut, SEQ_LUT, sizeof(SEQ_LUT));
	memcpy(C.sub_mat, SUB_MAT, sizeof(SUB_MAT));

	uhz num = in.num;
	uhz sum = in.meta[num - 1].off + in.meta[num - 1].len + 1;
	usz meta_bytes = bytesof(in.meta, num);

	dev letters, meta;
	CALL(cuMemAlloc(&letters, sum));
	CALL(cuMemAlloc(&meta, meta_bytes));
	CALL(cuMemcpyHtoD(letters, in.seqs, sum));
	CALL(cuMemcpyHtoD(meta, in.meta, meta_bytes));
	C.letters = (u8 *)letters;
	C.meta = (struct meta *)meta;

	shz *matrix = out.matrix;
	usz alignments = alignments((usz)num);
	constexpr usz batch_size = 64 << 20;
	usz free = 0, total = 0;
	CALL(cuMemGetInfo(&free, &total));
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

	CALL(cuMemcpyHtoD(cu.constants, &C, sizeof(C)));

	uhz active = 0;
	dev scores[2] = {};
	usz batch = 0, batch_last = 0, batch_done = 0, b_scores = 0;
	if (C.triangular) {
		batch = min(alignments, batch_size);
		b_scores = bytesof(matrix, batch);
		CALL(cuMemAlloc(&scores[0], b_scores));
		CALL(cuMemsetD32(scores[0], 0, batch));
		CALL(cuMemAlloc(&scores[1], b_scores));
		CALL(cuMemsetD32(scores[1], 0, batch));
	} else {
		batch = alignments;
		b_scores = bytesof(matrix, num * num);
		CALL(cuMemAlloc(&scores[0], b_scores));
		CALL(cuMemsetD32(scores[0], 0, (usz)num * num));
	}

	bool subsequent = false, syncing = false, matrix_copied = false;
	pinfo("Performing %zu pairwise alignments", alignments);
	ppercent(0, "Aligning sequences");
	bench_align_start();
	while (batch_done < alignments) {
		usz offset = batch_last;
		if (offset >= alignments) {
			if (subsequent) {
				CALL(cuCtxSynchronize());
				active = 1 - active;
			}
			goto cuda_results;
		}

		if (C.triangular) {
			if (offset + batch > alignments)
				batch = alignments - offset;
			if (!batch) {
				if (subsequent)
					CALL(cuCtxSynchronize());
				goto cuda_results;
			}
			if (subsequent) {
				CALL(cuCtxSynchronize());
				active = 1 - active;
			}
		}

		void *args[] = { &scores[active], &offset, &batch };
		CALL(cuLaunchKernel(cu.kernel, (batch + cu.bx - 1) / cu.bx, 1,
				    1, cu.bx, 1, 1, 0, cu.compute, args,
				    nullptr));
		batch_last += batch;
cuda_results:
		if (!C.triangular) {
			if (matrix_copied)
				goto cuda_progress;

			CALL(cuStreamSynchronize(cu.compute));
			if (matrix)
				CALL(cuMemcpyDtoH(matrix, scores[0], b_scores));

			batch_done = alignments;
			matrix_copied = true;
			goto cuda_progress;
		}

		if (batch_done >= alignments) {
			if (syncing) {
				CALL(cuStreamSynchronize(cu.memory));
				syncing = false;
			}
			goto cuda_progress;
		}

		if (syncing) {
			int res = cuStreamQuery(cu.memory);
			if (res == 600 /*NOT_READY*/)
				goto cuda_progress;
			CALL(res);
			syncing = false;
		}

		if (!subsequent && batch < alignments) {
			subsequent = true;
			goto cuda_progress;
		}

		usz n_scores = min(batch, alignments - batch_done);
		if (!n_scores)
			goto cuda_progress;

		if (subsequent) {
			if (matrix)
				CALL(cuMemcpyDtoHAsync(
					matrix + batch_done, scores[1 - active],
					bytesof(matrix, n_scores), cu.memory));
			syncing = true;
		} else {
			CALL(cuStreamSynchronize(cu.compute));
			if (matrix)
				CALL(cuMemcpyDtoH(matrix + batch_done,
						  scores[active],
						  bytesof(matrix, n_scores)));
		}
		batch_done += n_scores;
cuda_progress:
		pproportc(batch_done / alignments, "Aligning sequences");
	}

	bench_align_end();
	ppercent(100, "Aligning sequences");
	bench_align_print();

	cuMemFree(letters);
	cuMemFree(meta);
	cuMemFree(scores[0]);
	if (C.triangular)
		cuMemFree(scores[1]);

	return true;
ask_cuda:
	no_cuda = print_Yn("Would you like to switch to non-CUDA (CPU)?");
	if (no_cuda)
		return align_cpu(in, out);
	return false;
}

static void print_cuda_enabled(void)
{
	pinfom("CUDA: Enabled");
}

ARG_EXTERN(align);
ARG_EXTERN(compression);
ARG_EXTERN(threads);

ARGUMENT(disable_cuda) = {
	.opt = 'C',
	.lopt = "no-cuda",
	.help = "Disable CUDA (if available)",
	.set = &no_cuda,
	.validate_callback = init_cuda,
	.validate_phase = ARG_CALLBACK_IF_UNSET,
	.validate_order = ARG_ORDER_AFTER(ARG(align)),
	.action_callback = print_cuda_enabled,
	.action_phase = ARG_CALLBACK_IF_UNSET,
	.action_order = ARG_ORDER_AFTER(ARG(compression)),
	.help_order = ARG_ORDER_AFTER(ARG(threads)),
};
