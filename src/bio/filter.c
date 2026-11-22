#include <args.h>
#include <print.h>
#include <progress.h>
#include <stdatomic.h>
#include <string.h>

#include "bio/method.h"
#include "io/input.h"
#include "system/os.h"
#include "util/benchmark.h"
#include "util/macros.h"

static float threshold;

enum : u8 { STATE_LOST = 1, STATE_KEPT = 2 };

struct thread_filter {
	struct input in;
	_Atomic u8 *states;
	float threshold;
	alignas(64) _Atomic usz cnt;
};

static void filter_worker(void *p)
{
	struct thread_filter *t = p;
	usz total = t->in.num - 1;
begin:
	usz cur = atomic_fetch_add_explicit(&t->cnt, 1, memory_order_relaxed);
	if (cur >= total)
		return;

	uhz j = (uhz)(cur + 1);
	u8 res = STATE_KEPT;
	struct meta m1 = t->in.meta[j];
	const u8 *restrict s1 = t->in.seqs + m1.off;
	for (uhz i = 0; i < j; i++) {
		u8 state;
		while (!(state = atomic_load_explicit(&t->states[i],
						      memory_order_acquire)))
			__builtin_ia32_pause();
		if (state == STATE_LOST)
			continue;

		struct meta m2 = t->in.meta[i];
		const u8 *restrict s2 = t->in.seqs + m2.off;
		uhz ml = min(m1.len, m2.len);
		if (LEN_BAD(ml) || SEQ_BAD(s1) || SEQ_BAD(s2))
			unreachable_release();

		uhz matches = 0;
		for (uhz k = 0; k < ml; k++)
			matches += s1[k] == s2[k];
		if ((float)matches / (float)ml >= t->threshold) {
			res = STATE_LOST;
			break;
		}
	}

	atomic_store_explicit(&t->states[j], res, memory_order_release);
	progress_add(1);
	goto begin;
}

bool filter(struct input *in)
{
	if (threshold <= 0.0f)
		return true;

	uhz num = in->num;
	_Atomic u8 *states = calloc(num, sizeof(*states));
	if (!states) {
		perr("Out of memory during sequence filtering");
		return false;
	}
	atomic_init(&states[0], STATE_KEPT);

	struct thread_filter t = {
		.in = *in,
		.states = states,
		.threshold = threshold,
	};
	atomic_init(&t.cnt, 0);

	progress_start(num - 1, THREAD_NUM, "Filtering sequences");
	bench_filter_start();
	if (!parallelize(filter_worker, &t)) {
		perr("Out of memory during sequence filtering");
		return false;
	}
	progress_end();

	in->max = 0;
	in->num = 0;
	for (uhz read = 0, used = 0; read < num; read++) {
		if (atomic_load_explicit(&states[read], memory_order_relaxed) ==
		    STATE_LOST)
			continue;

		struct meta m = in->meta[read];
		if (used != m.off)
			memmove(in->seqs + used, in->seqs + m.off, m.len + 1);
		m.off = used;
		used += m.len + 1;
		in->meta[in->num++] = m;
		in->max = max(in->max, m.len);
	}
	free(states);
	bench_filter_end();

	pinfo("Filtered out %u sequences", num - in->num);
	if (in->num < SEQ_N_MIN) {
		perr("Not enough sequences: %u (min: %u)", in->num, SEQ_N_MIN);
		return false;
	}

	bench_filter_print();
	return true;
}

ARG_PARSE_F(parse_filter, float, , (val < 0.0f || val > 1.0f),
	    "Filter threshold must be between 0.0 and 1.0")

static void print_filter(void)
{
	if (threshold > 0.0f)
		pinfom("Filter threshold: %.1f%%", threshold * 100.0f);
	else
		pwarnm("Filter: Ignored");
}

ARG_EXTERN(gap_penalty);
ARG_EXTERN(list_matrices);

ARGUMENT(filter_threshold) = {
	.opt = 'f',
	.lopt = "filter",
	.help = "Filter sequences with similarity above threshold [0.0-1.0]",
	.param = "FLOAT",
	.param_req = ARG_PARAM_REQUIRED,
	.dest = &threshold,
	.parse_callback = parse_filter,
	.action_callback = print_filter,
	.action_phase = ARG_CALLBACK_IF_SET,
	.action_order = ARG_ORDER_AFTER(ARG(gap_penalty)),
	.help_order = ARG_ORDER_AFTER(ARG(list_matrices)),
};
