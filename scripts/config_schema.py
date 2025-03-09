from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
user_file = project_root / "include" / "user.h"

TOOLTIPS = {
    "MAX_CSV_LINE": "Maximum length of any line in CSV files (must be ≥32)",
    "MAX_SEQ_LEN": "Maximum length of any sequence (must be ≥1)",
    "GAP_PENALTY": "Penalty for gaps when aligning sequences",
    "BATCH_SIZE": "Number of sequences to process in each batch for multi-threaded mode",
    "READ_CSV_HEADER": """Input CSV Format Rules:
- One sequence per line
- Fixed number of columns 
- Additional columns preserved in output
- Names are arbitrary, only positions matter

Example: animal,sequence,data1,data2""",
    "READ_CSV_SEQ_POS": "Position of sequence column in input (0-based)",
    "READ_CSV_COLS": "Total number of columns in input CSV",
    "WRITE_CSV_HEADER": """Output CSV Format Rules:
- Each row is paired with the next row
- Output columns come in pairs (one from first row, one from second row) 
- Column pairs must be adjacent in header
- Required special columns: score, alignment
- All column names are arbitrary, only positions matter
- Number of columns = 2 * number of read columns + 2 (+ 4 if similarity analysis enabled)

Examples:
seq_a,seq_b,score,align,name_x,name_y,type1,type2
score,first,second,alignment,cat1,cat2,id_x,id_y
type_orig,type_comp,name1,name2,seq_first,seq_second,aligned_sequence,points
if similarity analysis enabled, add: similarity,matches,mismatches,gaps""",
    "WRITE_CSV_SEQ1_POS": "Position of first sequence in output",
    "WRITE_CSV_SCORE_POS": "Position of alignment score in output",
    "WRITE_CSV_ALIGN_POS": "Position of alignment result in output",
    "WRITE_CSV_MATCHES_POS": "Position of matches in output",
    "WRITE_CSV_MISMATCHES_POS": "Position of mismatches in output",
    "WRITE_CSV_GAPS_POS": "Position of number of gaps in output",
    "WRITE_CSV_SIMILARITY_POS": "Position of similarity in output",
    "WRITE_CSV_ALIGN_FMT": "Printf format for alignment (must contain two %s)",
    "INPUT_FILE": "Path to input file",
    "OUTPUT_FILE": "Path to output file",
    "MODE_MULTITHREAD": "Uncheck to disable multithreaded mode (singlethreaded mode will be used)",
    "SIMILARITY_ANALYSIS": "Enable similarity analysis (make sure to update the write header accordingly)",
    "MODE_WRITE": "Uncheck to disable writing to output CSV file",
}

DEFAULT_VALUES = {
    "MAX_CSV_LINE": "256",
    "MAX_SEQ_LEN": "64",
    "GAP_PENALTY": "-4",
    "BATCH_SIZE": "32768",
    "READ_CSV_HEADER": "sequence,label",
    "READ_CSV_SEQ_POS": "0",
    "READ_CSV_COLS": "2",
    "WRITE_CSV_HEADER": "sequence1,sequence2,label1,label2,score,alignment,matches,mismatches,gaps,similarity",
    "WRITE_CSV_SEQ1_POS": "0",
    "WRITE_CSV_SCORE_POS": "4",
    "WRITE_CSV_ALIGN_POS": "5",
    "WRITE_CSV_MATCHES_POS": "6",
    "WRITE_CSV_MISMATCHES_POS": "7",
    "WRITE_CSV_GAPS_POS": "8",
    "WRITE_CSV_SIMILARITY_POS": "9",
    "WRITE_CSV_ALIGN_FMT": "\"('%s', '%s')\"",
    "INPUT_FILE": str(Path(str(project_root / "datasets" / "avpdb.csv")).as_posix()),
    "OUTPUT_FILE": str(Path(str(project_root / "results" / "results.csv")).as_posix()),
}

DEFAULT_CHECKBOXES = {
    "MODE_MULTITHREAD": False,
    "SIMILARITY_ANALYSIS": True,
    "MODE_WRITE": True,
}

DISPLAY_NAMES = {
    "MAX_CSV_LINE": "Maximum CSV Line Length",
    "MAX_SEQ_LEN": "Maximum Sequence Length",
    "GAP_PENALTY": "Gap Penalty",
    "BATCH_SIZE": "Batch Size",
    "READ_CSV_HEADER": "Input CSV Header",
    "READ_CSV_SEQ_POS": "Sequence Column Position",
    "READ_CSV_COLS": "Number of Columns",
    "WRITE_CSV_HEADER": "Output CSV Header",
    "WRITE_CSV_SEQ1_POS": "First Sequence Position",
    "WRITE_CSV_SCORE_POS": "Score Column Position",
    "WRITE_CSV_ALIGN_POS": "Alignment Column Position",
    "WRITE_CSV_MATCHES_POS": "Matches Column Position",
    "WRITE_CSV_MISMATCHES_POS": "Mismatches Column Position",
    "WRITE_CSV_GAPS_POS": "Gaps Number Column Position",
    "WRITE_CSV_SIMILARITY_POS": "Similarity Column Position",
    "WRITE_CSV_ALIGN_FMT": "Alignment Format",
    "INPUT_FILE": "Input File",
    "OUTPUT_FILE": "Output File",
    "MODE_MULTITHREAD": "Enable Multithreaded Mode (faster for files larger than ~10k-100k lines)",
    "SIMILARITY_ANALYSIS": "Enable Similarity Analysis",
    "MODE_WRITE": "Enable Writing to CSV File (useful during development)",
}


def validate_config(fields, checkboxes):
    try:
        read_header = fields["READ_CSV_HEADER"].get().strip()
        if not read_header:
            return False, "Input Header cannot be empty"

        read_cols = read_header.count(",") + 1
        write_mode = checkboxes["MODE_WRITE"].get()

        numeric_rules = {
            "MAX_CSV_LINE": (32, "≥32"),
            "MAX_SEQ_LEN": (1, "≥1"),
            "BATCH_SIZE": (1, "≥1"),
            "GAP_PENALTY": (0, "<0", lambda x: x < 0),
            "READ_CSV_SEQ_POS": (
                read_cols,
                f"between 0 and {read_cols-1}",
                lambda x: 0 <= x < read_cols,
            ),
            "READ_CSV_COLS": (
                read_cols,
                f"equal to {read_cols}",
                lambda x: x == read_cols,
            ),
        }

        for key, rule in numeric_rules.items():
            try:
                val = int(fields[key].get())
                check = rule[2] if len(rule) > 2 else lambda x: x >= rule[0]
                if not check(val):
                    return False, f"{DISPLAY_NAMES[key]} must be {rule[1]}"
            except ValueError:
                return False, f"Invalid numeric value for {DISPLAY_NAMES[key]}"

        if write_mode:
            write_header = fields["WRITE_CSV_HEADER"].get().strip()
            if not write_header:
                return False, "Output Header cannot be empty"

            write_cols = write_header.count(",") + 1
            expected_cols = (
                2 * read_cols + 2 + 4 * checkboxes["SIMILARITY_ANALYSIS"].get()
            )
            if write_cols != expected_cols:
                return (
                    False,
                    f"Output must have {expected_cols} columns (found {write_cols})",
                )

            pos_fields = [
                "WRITE_CSV_SEQ1_POS",
                "WRITE_CSV_SCORE_POS",
                "WRITE_CSV_ALIGN_POS",
                "WRITE_CSV_MATCHES_POS",
                "WRITE_CSV_MISMATCHES_POS",
                "WRITE_CSV_GAPS_POS",
                "WRITE_CSV_SIMILARITY_POS",
            ]

            for field in pos_fields:
                try:
                    if (
                        int(fields[field].get()) == -1
                        and int(fields[field].get()) != -2
                    ):
                        return False, f"{DISPLAY_NAMES[field]} must be set"
                except ValueError:
                    return False, f"Invalid numeric value for {DISPLAY_NAMES[field]}"

            positions = [int(fields[k].get()) for k in pos_fields[:3]]

            if any(not 0 <= p < write_cols for p in positions):
                return False, "Column positions must be within output column range"

            if len(set(positions + [positions[0] + 1])) != 4:
                return False, "Output columns must have unique positions"

            if fields["WRITE_CSV_ALIGN_FMT"].get().count("%s") != 2:
                return (
                    False,
                    "Alignment format must contain exactly two %s placeholders",
                )

        input_path = fields["INPUT_FILE"].get()
        output_path = fields["OUTPUT_FILE"].get()

        if not Path(input_path).exists():
            return False, f"Input file does not exist: {input_path}"

        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"Cannot create output directory: {str(e)}"

        return True, None

    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"


def save_config(fields, checkboxes):
    try:
        with open(user_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if not line.strip().startswith("#define"):
                new_lines.append(line)
                continue

            name = line.split()[1]
            if name in fields:
                value = fields[name].get()

                if name.endswith("_FILE"):
                    value = str(Path(value).as_posix())
                if name.endswith("_HEADER"):
                    value = value.rstrip("\n") + "\\n"
                elif name.endswith("_FMT"):
                    value = value.replace('"', '\\"')

                if any(name.endswith(x) for x in ["_FILE", "_HEADER", "_FMT"]):
                    value = f'"{value}"'

                new_lines.append(f"#define {name} {value}\n")

            elif name in checkboxes:
                value = "1" if checkboxes[name].get() else "0"
                new_lines.append(f"#define {name} {value}\n")

            else:
                new_lines.append(line)

        with open(user_file, "w") as f:
            f.writelines(new_lines)

        return True, None

    except Exception as e:
        return False, f"Failed to save configuration: {e}"
