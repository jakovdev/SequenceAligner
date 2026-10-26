#include <args.h>
#include <print.h>
#include <progress.h>
#include <string.h>
#include <stdlib.h>

#include "bio/align.h"
#include "bio/kernels.cuh"
#include "io/input.h"
#include "io/output.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

constexpr usz batch_size = 64 << 20;

#ifdef _WIN32
#define CuDLL "nvcuda.dll"
#define CuFN __stdcall
#else
#define CuDLL "libcuda.so.1"
#define CuFN
#endif

typedef int CUresult;
typedef int CUdevice;
typedef uintptr_t CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef void(CuFN *CUhostFn)(void *);

constexpr unsigned CU_STREAM_DEFAULT = 0;
constexpr unsigned CU_EVENT_DISABLE_TIMING = 0x2;

#define cuCtxCreate cuCtxCreate_v2
#define cuModuleGetGlobal cuModuleGetGlobal_v2
#define cuMemGetInfo cuMemGetInfo_v2
#define cuMemAlloc cuMemAlloc_v2
#define cuMemcpyHtoD cuMemcpyHtoD_v2
#define cuMemcpyDtoH cuMemcpyDtoH_v2
#define cuMemcpyDtoHAsync cuMemcpyDtoHAsync_v2
#define cuMemsetD8 cuMemsetD8_v2
#define cuMemFree cuMemFree_v2
#define cuStreamDestroy cuStreamDestroy_v2
#define cuCtxDestroy cuCtxDestroy_v2
#define cuEventDestroy cuEventDestroy_v2

#define CU_FUNCS(X)                                                          \
	X(cuGetErrorString, (CUresult, const char **str))                    \
	X(cuInit, (unsigned flags))                                          \
	X(cuDeviceGetCount, (int *count))                                    \
	X(cuDeviceGet, (CUdevice *, int ordinal))                            \
	X(cuCtxCreate, (CUcontext *, unsigned flags, CUdevice))              \
	X(cuModuleLoadData, (CUmodule *, const void *data))                  \
	X(cuModuleGetFunction, (CUfunction *, CUmodule, const char *))       \
	X(cuModuleGetGlobal, (CUdeviceptr *, usz *, CUmodule, const char *)) \
	X(cuOccupancyMaxPotentialBlockSize,                                  \
	  (int *grid_min, int *blocks, CUfunction, void *, usz, int limit))  \
	X(cuStreamCreate, (CUstream *, unsigned flags))                      \
	X(cuEventCreate, (CUevent *, unsigned flags))                        \
	X(cuMemGetInfo, (usz * free, usz * total))                           \
	X(cuMemAlloc, (CUdeviceptr *, usz))                                  \
	X(cuMemcpyHtoD, (CUdeviceptr, const void *, usz))                    \
	X(cuMemcpyDtoH, (void *, CUdeviceptr, usz))                          \
	X(cuMemcpyDtoHAsync, (void *, CUdeviceptr, usz, CUstream))           \
	X(cuMemsetD8, (CUdeviceptr, unsigned char, usz))                     \
	X(cuLaunchKernel,                                                    \
	  (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,     \
	   unsigned, unsigned, CUstream, void **, void **))                  \
	X(cuLaunchHostFunc, (CUstream, CUhostFn, void *))                    \
	X(cuCtxSynchronize, (void))                                          \
	X(cuStreamWaitEvent, (CUstream, CUevent, unsigned flags))            \
	X(cuEventRecord, (CUevent, CUstream))                                \
	X(cuMemFree, (CUdeviceptr))                                          \
	X(cuStreamDestroy, (CUstream))                                       \
	X(cuEventDestroy, (CUevent))                                         \
	X(cuModuleUnload, (CUmodule))                                        \
	X(cuCtxDestroy, (CUcontext))

#define X(name, args) static CuFN CUresult(*name) args;
CU_FUNCS(X)
#undef X

#define STR(FN) #FN
#define CALL(function)                               \
	do {                                         \
		CUresult err = (function);           \
		if (err) {                           \
			const char *msg;             \
			cuGetErrorString(err, &msg); \
			perr("CUDA: %s", msg);       \
			goto cuda_error;             \
		}                                    \
	} while (0)

static struct gpu_nvidia {
	void *dll;
	CUcontext context;
	CUmodule module;
	CUfunction kernel;
	CUstream compute;
	CUstream memory;
	CUevent launch[2];
	CUevent copy[2];
	CUdeviceptr constants;
	CUdevice device;
	int bx;
} cu;

bool cuda_disable;

static void cuda_free(void)
{
	cuEventDestroy(cu.copy[1]);
	cuEventDestroy(cu.copy[0]);
	cuEventDestroy(cu.launch[1]);
	cuEventDestroy(cu.launch[0]);
	cuStreamDestroy(cu.memory);
	cuStreamDestroy(cu.compute);
	cuModuleUnload(cu.module);
	cuCtxDestroy(cu.context);
	dll_close(cu.dll);
	memset(&cu, 0, sizeof(cu));
}

static struct arg_callback cuda_init(void)
{
	if (!ALIGN)
		return ARG_VALID();

	cu.dll = dll_open(CuDLL);
	if (!cu.dll)
		goto cuda_not_found;

#define SYM(FN)                                \
	do {                                   \
		FN = dll_sym(cu.dll, STR(FN)); \
		if (!FN) {                     \
			dll_close(cu.dll);     \
			cu.dll = nullptr;      \
			goto cuda_not_found;   \
		}                              \
	} while (0)

#define X(name, args) SYM(name);
	CU_FUNCS(X)
#undef X

	int count = 0;
	if (cuInit(0) || cuDeviceGetCount(&count) || !count) {
		pwarn("No CUDA Devices found, defaulting to CPU-only");
		goto cuda_not_found;
	}

	/* TODO: Allow Multi-Device Execution */
	CALL(cuDeviceGet(&cu.device, 0));
	CALL(cuCtxCreate(&cu.context, 0, cu.device));
	static const unsigned char kernels[] = {
#embed "../generated/kernels.fatbin"
	};
	CALL(cuModuleLoadData(&cu.module, kernels));
	CALL(cuModuleGetFunction(&cu.kernel, cu.module, ALIGN->kernel));
	usz constants_size = 0;
	CALL(cuModuleGetGlobal(&cu.constants, &constants_size, cu.module, "C"));
	if (constants_size != sizeof(struct constants)) {
		perr("CUDA Kernel got corrupted, please report this");
		goto cuda_error;
	}
	int grid_min = 0;
	CALL(cuOccupancyMaxPotentialBlockSize(&grid_min, &cu.bx, cu.kernel,
					      nullptr, 0, 0));
	CALL(cuStreamCreate(&cu.compute, CU_STREAM_DEFAULT));
	CALL(cuStreamCreate(&cu.memory, CU_STREAM_DEFAULT));
	CALL(cuEventCreate(&cu.launch[0], CU_EVENT_DISABLE_TIMING));
	CALL(cuEventCreate(&cu.launch[1], CU_EVENT_DISABLE_TIMING));
	CALL(cuEventCreate(&cu.copy[0], CU_EVENT_DISABLE_TIMING));
	CALL(cuEventCreate(&cu.copy[1], CU_EVENT_DISABLE_TIMING));
	atexit(cuda_free);
	return ARG_VALID();
cuda_error:
	if (!print_Yn("Would you like to switch to non-CUDA (CPU)?"))
		return ARG_INVALID("You can also pass --no-cuda");
cuda_not_found:
	cuda_disable = true;
	return ARG_VALID();
}

usz memory_gpu(void)
{
	if (cuda_disable)
		return 0;
	usz free = 0, total = 0;
	CALL(cuMemGetInfo(&free, &total));
	return free;
cuda_error:
	cuda_disable = print_Yn("Would you like to switch to non-CUDA (CPU)?");
	if (!cuda_disable)
		exit(EXIT_FAILURE);
	return 0;
}

static void CuFN cuda_progress(void *amount)
{
	progress_add((usz)amount);
}

bool align_cuda(struct input in, struct output out)
{
	bool success = false;
	uhz buffers = 1;
	CUdeviceptr letters = {}, meta = {}, scores[2] = {};

	if (in.max > MAX_CUDA_SEQUENCE_LENGTH) {
		perr("Sequence length exceeds CUDA Device limits");
		goto cuda_error;
	}

	struct constants C = {
		.num = in.num,
		.gap_pen = GAP_PEN,
		.gap_opn = GAP_OPN,
		.gap_ext = GAP_EXT,
	};

	memcpy(C.seq_lut, SEQ_LUT, sizeof(SEQ_LUT));
	memcpy(C.sub_mat, SUB_MAT, sizeof(SUB_MAT));

	usz num = in.num;
	uhz sum = in.meta[num - 1].off + in.meta[num - 1].len + 1;

	CALL(cuMemAlloc(&letters, sum));
	CALL(cuMemAlloc(&meta, bytesof(in.meta, num)));
	CALL(cuMemcpyHtoD(letters, in.seqs, sum));
	CALL(cuMemcpyHtoD(meta, in.meta, bytesof(in.meta, num)));
	C.letters = (u8 *)letters;
	C.meta = (struct meta *)meta;

	usz free = 0, total = 0;
	CALL(cuMemGetInfo(&free, &total));
	pverb("GPU: %.2f GiB free / %.2f GiB total", (double)free / (double)GiB,
	      (double)total / (double)GiB);
	free = free * 3 / 4;

	shz *matrix = out.matrix;
	usz alignments = alignments(num);
	usz stride = alignments;
	usz scores_count = num * num;

	if (out.triangular || bytesof(matrix, scores_count) > free) {
		scores_count = min(batch_size, alignments);
		if (bytesof(matrix, scores_count * 2) > free) {
			perr("Not enough CUDA Device memory for alignment");
			goto cuda_error;
		}
		C.triangular = true;
		stride = batch_size;
		buffers = 2;
	}

	CALL(cuMemcpyHtoD(cu.constants, &C, sizeof(C)));

	for (uhz b = 0; b < buffers; b++) {
		CALL(cuMemAlloc(&scores[b], bytesof(matrix, scores_count)));
		CALL(cuMemsetD8(scores[b], 0, bytesof(matrix, scores_count)));
		CALL(cuEventRecord(cu.copy[b], cu.compute));
	}

	pinfo("Performing %zu pairwise alignments", alignments);
	progress_start(alignments, 1, "Aligning sequences");
	bench_align_start();

	uhz active = 0;
	for (usz offset = 0, batch = 0; offset < alignments; offset += batch) {
		batch = min(stride, alignments - offset);

		CUstream busy = cu.compute;
		CALL(cuStreamWaitEvent(busy, cu.copy[active], 0));

		void *args[] = { &scores[active], &offset, &batch };
		CALL(cuLaunchKernel(cu.kernel, (batch + cu.bx - 1) / cu.bx, 1,
				    1, cu.bx, 1, 1, 0, busy, args, nullptr));
		CALL(cuLaunchHostFunc(busy, cuda_progress, (void *)batch));
		CALL(cuEventRecord(cu.launch[active], busy));

		if (matrix) {
			busy = cu.memory;
			usz count = C.triangular ? batch : scores_count;
			CALL(cuStreamWaitEvent(busy, cu.launch[active], 0));
			CALL(cuMemcpyDtoHAsync(matrix + offset, scores[active],
					       bytesof(matrix, count), busy));
		}

		CALL(cuEventRecord(cu.copy[active], busy));
		active ^= C.triangular;
	}

	CALL(cuCtxSynchronize());
	bench_align_end();
	success = true;
cuda_error:
	if (!success)
		cuCtxSynchronize();
	progress_end();
	for (uhz b = 0; b < buffers; b++) {
		if (scores[b])
			cuMemFree(scores[b]);
	}
	if (letters)
		cuMemFree(letters);
	if (meta)
		cuMemFree(meta);
	if (success) {
		bench_align_print();
		return true;
	}

	cuda_disable = print_Yn("Would you like to switch to non-CUDA (CPU)?");
	return cuda_disable ? align_cpu(in, out) : false;
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
	.set = &cuda_disable,
	.validate_callback = cuda_init,
	.validate_phase = ARG_CALLBACK_IF_UNSET,
	.validate_order = ARG_ORDER_AFTER(ARG(align)),
	.action_callback = print_cuda_enabled,
	.action_phase = ARG_CALLBACK_IF_UNSET,
	.action_order = ARG_ORDER_AFTER(ARG(compression)),
	.help_order = ARG_ORDER_AFTER(ARG(threads)),
};
