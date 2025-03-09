#include "csv.h"

#if MODE_MULTITHREAD == 1    
#include "thread.h"
#endif // MODE_MULTITHREAD

int main(void) {
    SET_HIGH_CLASS();
    #if MODE_MULTITHREAD == 1
    init_thread_pool();
    #else // MODE_MULTITHREAD == 0
    PIN_THREAD(0);
    #endif // MODE_MULTITHREAD
    
    Files files = get_files();
    char* current = files.file_data;
    char* end = files.file_data + files.data_size;
    current = skip_header(current, end);
    
    init_format();
    ScoringMatrix scoring;
    init_scoring_matrix(&scoring);

    #if MODE_MULTITHREAD == 1
    Sequence* seqs = (Sequence*)malloc(sizeof(Sequence) * BATCH_SIZE);
    OtherData* other = (OtherData*)malloc(sizeof(OtherData) * BATCH_SIZE);
    size_t* seq_lens = (size_t*)malloc(sizeof(size_t) * BATCH_SIZE);
    size_t seq_count = 1;

    double start = get_time();

    seq_lens[0] = parse_csv_line(&current, seqs[0].data, other[0].data);
    while (current < end && *current) {
        while (seq_count < BATCH_SIZE && current < end && *current) {
            seq_lens[seq_count] = parse_csv_line(&current, seqs[seq_count].data, other[seq_count].data);
            seq_count++;
        }

        size_t num_pairs = seq_count - 1;
        AlignTask* tasks = (AlignTask*)malloc(sizeof(AlignTask) * num_pairs);
        Alignment* results = (Alignment*)malloc(sizeof(Alignment) * num_pairs);

        for (size_t i = 0; i < num_pairs; i++) {
            tasks[i] = (AlignTask){
                .seq1 = seqs[i].data,
                .seq2 = seqs[i + 1].data,
                .len1 = seq_lens[i],
                .len2 = seq_lens[i + 1],
                .scoring = &scoring,
                .result = &results[i]
            };
        }

        size_t tasks_per_thread = num_pairs / g_num_threads;
        for (int t = 0; t < g_num_threads; t++) {
            g_thread_work[t].tasks = tasks;
            g_thread_work[t].start = t * tasks_per_thread;
            g_thread_work[t].end = (t == g_num_threads - 1) ? num_pairs : (t + 1) * tasks_per_thread;
            sem_post(g_thread_work[t].work_ready);
        }

        for (int t = 0; t < g_num_threads; t++) sem_wait(g_thread_work[t].work_done);

        #if MODE_WRITE == 1
        for (size_t i = 0; i < num_pairs; i++) {
            if (files.writer.pos >= WRITE_BUF - MAX_CSV_LINE * 2) {
                flush_buffer(&files.writer);
            }

            Data prev = {seqs[i].data, other[i].data, seq_lens[i]};
            Data curr = {seqs[i + 1].data, other[i + 1].data, seq_lens[i + 1]};
            files.writer.pos += buffer_output(files.writer.buffer, files.writer.pos, &prev, &curr, &results[i]);
        }
        #endif // MODE_WRITE

        // Keep last sequence for next batch
        strcpy(seqs[0].data, seqs[seq_count - 1].data);
        strcpy(other[0].data, other[seq_count - 1].data);
        seq_lens[0] = seq_lens[seq_count - 1];
        seq_count = 1;

        free(tasks);
        free(results);
    }

    free(seqs);
    free(other);
    free(seq_lens);
    destroy_thread_pool();

    #else // MODE_MULTITHREAD == 0

    char seq[MAX_SEQ_LEN];
    char prev_seq[MAX_SEQ_LEN];
    char data[MAX_CSV_LINE - MAX_SEQ_LEN];
    char prev_data[MAX_CSV_LINE - MAX_SEQ_LEN];

    double start = get_time();
    size_t prev_len = parse_csv_line(&current, prev_seq, prev_data);
    while (current < end && *current) {
        size_t curr_len = parse_csv_line(&current, seq, data);
        Alignment result = align_sequences(prev_seq, prev_len, seq, curr_len, &scoring);

        #if MODE_WRITE == 1
        if (files.writer.pos >= WRITE_BUF - MAX_CSV_LINE * 2) {
            flush_buffer(&files.writer);
        }
        Data prev = {prev_seq, prev_data, prev_len};
        Data curr = {seq, data, curr_len};
        files.writer.pos += buffer_output(files.writer.buffer, files.writer.pos, &prev, &curr, &result);
        #else // MODE_WRITE == 0
        if (result.score < -1000000000) {
            // Will never happen but prevents compiler from removing unused result when not writing
            printf("Unexpected score (-1000000000)!\n");
        }
        #endif // MODE_WRITE

        strcpy(prev_data, data);
        strcpy(prev_seq, seq);
        prev_len = curr_len;
    }

    #endif // MODE_MULTITHREAD

    #if MODE_WRITE == 1
    flush_buffer(&files.writer);
    #endif // MODE_WRITE

    double endt = get_time();

    free_files(&files);
    
    printf("Alignment time: %f seconds\n", endt - start);
    return 0;
}