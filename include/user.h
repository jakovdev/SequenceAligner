#ifndef USER_H
#define USER_H

// Note: The provided values are defaults and can be changed.

// The maximum length of any line in your CSV file
#define MAX_CSV_LINE 256

// The maximum length of any sequence in your CSV file
#define MAX_SEQ_LEN 64

// The gap penalty for the alignment
#define GAP_PENALTY -4

/* INPUT CSV FORMAT RULES
 * - One sequence per line
 * - Fixed number of columns
 * - Additional columns preserved in output
 * - Names are arbitrary, only positions matter
 * 
 * Example: "animal,sequence,data1,data2" */
#define READ_CSV_HEADER "sequence,label\n"

// Position of sequence column (0-based)
#define READ_CSV_SEQ_POS 0

// Number of columns in the CSV file
#define READ_CSV_COLS 2

/* OUTPUT CSV FORMAT RULES
 * - Each row is paired with the next row
 * - Output columns come in pairs (one from first row, one from second row)
 * - Column pairs must be adjacent in the header
 * - Required special columns: score, alignment
 * - All column names are arbitrary, only positions matter
 * - Number of columns = 2 * number of read columns + 2
 * 
 * Valid examples:
 * "seq_a,seq_b,score,align,name_x,name_y,type1,type2"
 * "score,first,second,alignment,cat1,cat2,id_x,id_y"
 * "type_orig,type_comp,name1,name2,seq_first,seq_second,aligned_sequence,points" 
 * if doing similarity analysis, add matches, mismatches, gaps, similarity columns
 */
#define WRITE_CSV_HEADER "sequence1,sequence2,label1,label2,score,alignment,matches,mismatches,gaps,similarity\n"

// Sequence 2 position is automatically calculated (this + 1)
#define WRITE_CSV_SEQ1_POS 0
#define WRITE_CSV_SCORE_POS 4
#define WRITE_CSV_ALIGN_POS 5
#define WRITE_CSV_MATCHES_POS 6
#define WRITE_CSV_MISMATCHES_POS 7
#define WRITE_CSV_GAPS_POS 8
#define WRITE_CSV_SIMILARITY_POS 9

// Alignment format for printf, only rules are to have two %s and to follow printf syntax
#define WRITE_CSV_ALIGN_FMT "\"('%s', '%s')\""

// Paths must be absolute. You can populate these with the python user script, or just copy paste the desired absolute paths.
#define INPUT_FILE
#define OUTPUT_FILE

// Modes //
#define MODE_MULTITHREAD 0
#define SIMILARITY_ANALYSIS 1
#define MODE_WRITE 1

// Speed constants //
#define BATCH_SIZE 32768

// Helper constants, do not change //
#define KiB (1ULL << 10)
#define MiB (KiB  << 10)
#define GiB (MiB  << 10)

#define WRITE_BUF (128 * KiB) // Buffer size for writing output
#endif