#ifndef USER_H
#define USER_H

// These will be made runtime later
// NOTE: THIS FILE IS TEMPORARY

// The maximum length of any line in your CSV file
#define MAX_CSV_LINE 256

// The maximum length of any sequence in your CSV file
#define MAX_SEQ_LEN 64

// Will be moved to skip_header////////////
// Position of sequence column (0-based)
#define READ_CSV_SEQ_POS 0
// Number of columns in the CSV file
#define READ_CSV_COLS 2
///////////////////////////////////////////

// Creates new strings with '-' for gaps, 0 == score only is calculated
#define MODE_CREATE_ALIGNED_STRINGS 0

// Helper constants, do not change //
#define KiB (1ULL << 10)
#define MiB (KiB  << 10)
#define GiB (MiB  << 10)

// Speed constants
#define LARGE_MATRIX_THRESHOLD (8 * MiB) // Threshold for using huge pages
#endif