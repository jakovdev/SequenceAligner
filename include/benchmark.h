#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "common.h"
#include "print.h"

typedef struct {
    double init_start;
    double init;
    double elapsed_init_time;
    double align_start;
    double align;
    double write_start;
    double write;
    double total;
    size_t total_alignments;
} BenchmarkTimes;

static BenchmarkTimes g_times = {0};

INLINE void bench_init_start(void) {
    g_times.init_start = get_time();
    g_times.elapsed_init_time = 0.0;
}

INLINE void bench_init_end(void) {
    if (get_mode_benchmark()) {
        g_times.init = get_time() - g_times.init_start - g_times.elapsed_init_time;
        print_timing("Initialization: %.3f sec", g_times.init);
    }
}

INLINE double bench_pause_init(void) {
    double current_time = get_time();
    g_times.elapsed_init_time += (current_time - g_times.init_start);
    return current_time;
}

INLINE void bench_resume_init(double saved_time) {
    g_times.init_start = get_time() - (saved_time - g_times.init_start) + g_times.elapsed_init_time;
}

INLINE void bench_align_start(void) {
    g_times.align_start = get_time();
}

INLINE void bench_align_end(void) {
    if (get_mode_benchmark()) {
        g_times.align = get_time() - g_times.align_start;
        print_timing("Computation: %.3f sec", g_times.align);
    }
}

INLINE void bench_write_start(void) {
    g_times.write_start = get_time();
}

INLINE void bench_write_end(void) {
    if (get_mode_benchmark()) {
        g_times.write = get_time() - g_times.write_start;
        print_timing("I/O operations: %.3f sec", g_times.write);
    }
}

INLINE void bench_set_alignments(size_t total_alignments) {
    g_times.total_alignments = total_alignments;
}

INLINE void bench_total(void) {
    if (message_config.quiet) return;
    if (get_mode_benchmark()) {
        g_times.total = g_times.init + g_times.align + g_times.write;
        print_step_header_start("Performance Summary");
        print_timing("Timing breakdown:");
        char init_str[32];
        snprintf(init_str, sizeof(init_str), "%.3f sec (%.1f%%)", g_times.init, (g_times.init / g_times.total) * 100);
        
        apply_box_color();
        printf("%s", BOX_VERTICAL);
        reset_color();
        
        printf(" %s%s %s: %s", ANSI_COLOR_CYAN, BOX_TEE_RIGHT, "Init", init_str);
        
        // TODO: Fix me (numbers match but in wrong place)
        int padding = OUTPUT_WIDTH + 3 - strlen(" ├─ Init: ") - strlen(init_str);
        for (int i = 0; i < padding; i++) printf(" ");
        
        apply_box_color();
        printf("%s\n", BOX_VERTICAL);
        reset_color();
        
        char align_str[32];
        snprintf(align_str, sizeof(align_str), "%.3f sec (%.1f%%)", g_times.align, (g_times.align / g_times.total) * 100);
        
        apply_box_color();
        printf("%s", BOX_VERTICAL);
        reset_color();
        
        printf(" %s%s %s: %s", ANSI_COLOR_CYAN, BOX_TEE_RIGHT, "Compute", align_str);
        
        // TODO: Fix me
        padding = OUTPUT_WIDTH + 3 - strlen(" ├─ Compute: ") - strlen(align_str);
        for (int i = 0; i < padding; i++) printf(" ");
        
        apply_box_color();
        printf("%s\n", BOX_VERTICAL);
        reset_color();
        
        char write_str[32];
        snprintf(write_str, sizeof(write_str), "%.3f sec (%.1f%%)", g_times.write, (g_times.write / g_times.total) * 100);
        
        apply_box_color();
        printf("%s", BOX_VERTICAL);
        reset_color();
        
        printf(" %s%s %s: %s", ANSI_COLOR_CYAN, BOX_TEE_RIGHT, "I/O", write_str);
        
        // TODO: Fix me
        padding = OUTPUT_WIDTH + 3 - strlen(" ├─ I/O: ") - strlen(write_str);
        for (int i = 0; i < padding; i++) printf(" ");
        
        apply_box_color();
        printf("%s\n", BOX_VERTICAL);
        reset_color();
        
        char total_str[32];
        snprintf(total_str, sizeof(total_str), "%.3f sec (100.0%%)", g_times.total);
        
        apply_box_color();
        printf("%s", BOX_VERTICAL);
        reset_color();
        
        printf(" %s%s %s: %s", ANSI_COLOR_CYAN, BOX_BOTTOM_LEFT, "Total", total_str);
        
        padding = OUTPUT_WIDTH + 3 - strlen(" └─ Total: ") - strlen(total_str);
        for (int i = 0; i < padding; i++) printf(" ");
        
        apply_box_color();
        printf("%s\n", BOX_VERTICAL);
        reset_color();
        
        double alignments_per_sec = g_times.total_alignments / g_times.align;
        print_timing("Alignments per second: %.2f", alignments_per_sec);
        
        if (get_num_threads() > 1) {
            double avg_alignment_time = g_times.align / get_num_threads();
            print_timing("Average time per thread: %.3f sec", avg_alignment_time);
            print_timing("Alignments per second per thread: %.2f", alignments_per_sec / get_num_threads());
        }
    }
}

#endif