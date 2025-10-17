#include "core/app/args.h"
#include "core/app/thread.h"
#include "core/bio/score/scoring.h"
#include "core/interface/seqalign_cuda.h"
#include "core/interface/seqalign_hdf5.h"
#include "system/arch.h"
#include "util/benchmark.h"
#include "util/print.h"

int
main(int argc, char* argv[])
{
    args_init(argc, argv);

    print(HEADER, MSG_NONE, "SEQUENCE ALIGNER");

    args_print_config();

    print(SECTION, MSG_NONE, "Setting Up Alignment");

    if (!args_mode_multithread())
    {
        PIN_THREAD(0);
    }

    bench_io_start();

    if (!sequences_load_from_file())
    {
        return 1;
    }

    bench_io_end();

    sequence_count_t sequence_count = sequences_count();
    alignment_size_t total_alignments = sequences_alignment_count();

#ifdef USE_CUDA
    if (args_mode_cuda() && !cuda_init())
    {
        return 1;
    }

#endif

    print_error_prefix("HDF5");

    bench_io_start();
    if (!h5_open(args_output(), sequence_count, args_compression(), args_mode_write()))
    {
        bench_io_end();
        print(ERROR, MSG_NONE, "Failed to create file, will use no-write mode");

        if (!print_yN("Do you want to continue? [y/N]"))
        {
            print(INFO, MSG_LOC(LAST), "Exiting due to file creation failure");
            h5_close(1);
            return 1;
        }

        bench_io_start();
        h5_open(NULL, 0, 0, false);
    }

    if (!h5_sequences_store(sequences_get(), sequence_count))
    {
        bench_io_end();
        print(ERROR, MSG_NONE, "Failed to store sequences, will use no-write mode");

        if (!print_yN("Do you want to continue? [y/N]"))
        {
            print(INFO, MSG_LOC(LAST), "Exiting due to sequence store failure");
            h5_close(1);
            return 1;
        }

        bench_io_start();
        h5_open(NULL, 0, 0, false);
    }

    bench_io_end();

    print(VERBOSE, MSG_LOC(FIRST), "Initializing substitution matrix");
    scoring_matrix_init();

    print(SECTION, MSG_NONE, "Performing Alignments");

    print(INFO, MSG_NONE, "Will perform %zu pairwise alignments", total_alignments);

    if (!align())
    {
        print(ERROR, MSG_NONE, "Failed to perform alignments");
        h5_close(1);
        return 1;
    }

    if (!args_mode_write())
    {
        print(INFO, MSG_NONE, "Matrix checksum: %lld", h5_checksum());
    }

    bench_print_align();

    bench_io_start();
    h5_close(0);
    bench_io_end();

    bench_print_io();

    bench_print_total(total_alignments);
    return 0;
}