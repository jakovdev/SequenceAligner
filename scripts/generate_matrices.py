import parasail
import os
import sys
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

if os.path.exists("scripts/generate_matrices.py"):
    OUTPUT_FILE = "include/matrices.h"
else:
    OUTPUT_FILE = "../include/matrices.h"

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
NUCLEOTIDES = "ACGT"

MATRIX_TYPES = {
    "amino": {
        "alphabet": AMINO_ACIDS,
        "matrices": [],
        "pattern": r"(blosum\d+|pam\d+)",
        "id": "SEQ_TYPE_AMINO",
        "name": "Amino acids",
        "description": "protein sequences",
        "aliases": ["amino", "aa", "protein"],
    },
    "nucleotide": {
        "alphabet": NUCLEOTIDES,
        "matrices": [],
        "pattern": r"(dnafull|nuc44)",
        "id": "SEQ_TYPE_NUCLEOTIDE",
        "name": "Nucleotides",
        "description": "DNA/RNA sequences",
        "aliases": ["nucleotide", "dna", "rna", "nt"],
    },
}


def get_available_matrices():
    matrix_attributes = [
        attr
        for attr in dir(parasail)
        if (re.match(r"blosum\d+|pam\d+|dnafull|nuc44", attr))
        and not attr.startswith("__")
    ]

    logger.info(f"Found {len(matrix_attributes)} potential matrices in parasail")

    for name in matrix_attributes:
        try:
            matrix = getattr(parasail, name)
            matrix_type = None

            # Determine matrix type by name pattern
            for type_name, type_info in MATRIX_TYPES.items():
                if re.match(type_info["pattern"], name):
                    matrix_type = type_name
                    break

            if not matrix_type:
                continue

            # Wrapper to provide the get_value method
            class ParasailMatrixWrapper:
                def __init__(self, matrix_obj, matrix_name, alphabet):
                    self.matrix_obj = matrix_obj
                    self.name = matrix_name
                    self.alphabet = alphabet
                    self.size = len(alphabet)

                def get_value(self, a, b):
                    try:
                        if hasattr(self.matrix_obj, "matrix"):
                            i = self.alphabet.find(a)
                            j = self.alphabet.find(b)
                            if (
                                i >= 0
                                and j >= 0
                                and i < len(self.matrix_obj.matrix)
                                and j < len(self.matrix_obj.matrix[0])
                            ):
                                return self.matrix_obj.matrix[i][j]
                        return 0
                    except Exception:
                        return 0

            alphabet = MATRIX_TYPES[matrix_type]["alphabet"]
            wrapped_matrix = ParasailMatrixWrapper(matrix, name, alphabet)

            # Test
            wrapped_matrix.get_value(alphabet[0], alphabet[0])

            MATRIX_TYPES[matrix_type]["matrices"].append((name.upper(), wrapped_matrix))

        except (AttributeError, TypeError):
            pass

    for type_name, type_info in MATRIX_TYPES.items():
        logger.info(
            f"Successfully loaded {len(type_info['matrices'])} {type_name} matrices"
        )

    return MATRIX_TYPES


def format_matrix_as_c_array(matrix):
    formatted_rows = []
    alphabet = matrix.alphabet

    max_width = 1
    for aa1 in alphabet:
        for aa2 in alphabet:
            val = matrix.get_value(aa1, aa2)
            width = len(str(val))
            max_width = max(max_width, width)

    # Format with consistent spacing
    for aa1 in alphabet:
        row_values = []
        for aa2 in alphabet:
            val = matrix.get_value(aa1, aa2)
            formatted_val = f"{val:>{max_width}}"
            row_values.append(formatted_val)

        formatted_row = "    {" + ", ".join(row_values) + "}, // " + aa1
        formatted_rows.append(formatted_row)

    return "{\n" + "\n".join(formatted_rows) + "\n    }"


def generate_matrix_tables():
    matrix_types = get_available_matrices()
    header_content = [
        "#ifndef MATRICES_H",
        "#define MATRICES_H",
        "",
        '#include "macros.h"',
        "",
        "typedef enum {",
    ]

    for i, (type_name, type_info) in enumerate(matrix_types.items()):
        if not type_info["matrices"]:
            continue
        header_content.append(f"    {type_info['id']} = {i},")

    header_content.extend(
        [
            "    // NOTE: This enum is kept minimal by design. Only standard biological sequence types",
            "    //       are included as they're the only ones with established scoring matrices.",
            "    SEQ_TYPE_COUNT",
            "} SequenceType;",
            "",
        ]
    )

    header_content.extend(
        [
            "typedef struct {",
            "    SequenceType type;",
            "    const char* name;",
            "    const char* description;",
            "    const char** aliases;",
            "} SequenceTypeInfo;",
            "",
            "static const SequenceTypeInfo SEQUENCE_TYPES[] = {",
        ]
    )

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue
        aliases = ", ".join([f'"{alias}"' for alias in type_info["aliases"]]) + ", NULL"
        header_content.append(f"    {{")
        header_content.append(f'        {type_info["id"]},')
        header_content.append(f'        "{type_info["name"]}",')
        header_content.append(f'        "{type_info["description"]}",')
        header_content.append(f"        (const char*[]){{{aliases}}}")
        header_content.append(f"    }},")

    header_content.append("};")
    header_content.append("")

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet = type_info["alphabet"]
        alphabet_name = type_name.upper()
        header_content.extend(
            [
                f'static const char {alphabet_name}_ALPHABET[] = "{alphabet}";',
                f"#define {alphabet_name}_SIZE {len(alphabet)}",
                "",
            ]
        )

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet_name = type_name.upper()
        type_info["matrices"].sort()

        header_content.append(
            f"#define NUM_{alphabet_name}_MATRICES {len(type_info['matrices'])}"
        )

        header_content.extend(
            [
                "",
                f"typedef struct {{",
                f"    const char* name;",
                f"    const int (*matrix)[{alphabet_name}_SIZE];",
                f"}} {type_name.capitalize()}Matrix;",
                "",
            ]
        )

        header_content.append(f"// {type_name.capitalize()} matrix identifiers")
        header_content.append(f"typedef enum {{")
        for i, (name, _) in enumerate(type_info["matrices"]):
            header_content.append(f"    {name}_ID = {i},")
        header_content.append(f"}} {type_name.capitalize()}MatrixID;")
        header_content.append("")

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet_name = type_name.upper()
        header_content.append(f"// {type_name.capitalize()} matrices")

        for name, matrix_obj in type_info["matrices"]:
            header_content.append(
                f"static const int {name}[{alphabet_name}_SIZE][{alphabet_name}_SIZE] = {format_matrix_as_c_array(matrix_obj)};"
            )
            header_content.append("")

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet_name = type_name.upper()
        header_content.append(
            f"static const {type_name.capitalize()}Matrix ALL_{alphabet_name}_MATRICES[NUM_{alphabet_name}_MATRICES] = {{"
        )
        for name, _ in type_info["matrices"]:
            header_content.append(f'    {{"{name}", {name}}},')
        header_content.append("};")
        header_content.append("")

    header_content.extend(
        [
            "INLINE const char* get_matrix_name_by_id(int seq_type, int matrix_id) {",
            "    if (seq_type < 0 || matrix_id < 0) {",
            '        return "Unknown";',
            "    }",
            "    ",
            "    if (seq_type == SEQ_TYPE_AMINO && matrix_id < NUM_AMINO_MATRICES) {",
            "        return ALL_AMINO_MATRICES[matrix_id].name;",
            "    } else if (seq_type == SEQ_TYPE_NUCLEOTIDE && matrix_id < NUM_NUCLEOTIDE_MATRICES) {",
            "        return ALL_NUCLEOTIDE_MATRICES[matrix_id].name;",
            "    }",
            '    return "Unknown";',
            "}",
            "",
            "INLINE int find_matrix_by_name(int seq_type, const char* name) {",
            "    if (!name) return -1;",
            "",
            "    int num_matrices = 0;",
            "    const void* matrices = NULL;",
            "",
            "    if (seq_type == SEQ_TYPE_AMINO) {",
            "        num_matrices = NUM_AMINO_MATRICES;",
            "        matrices = ALL_AMINO_MATRICES;",
            "    } else if (seq_type == SEQ_TYPE_NUCLEOTIDE) {",
            "        num_matrices = NUM_NUCLEOTIDE_MATRICES;",
            "        matrices = ALL_NUCLEOTIDE_MATRICES;",
            "    } else {",
            "        return -1;",
            "    }",
            "",
            "    for (int i = 0; i < num_matrices; i++) {",
            "        const char* matrix_name = NULL;",
            "        if (seq_type == SEQ_TYPE_AMINO) {",
            "            matrix_name = ((const AminoMatrix*)matrices)[i].name;",
            "        } else {",
            "            matrix_name = ((const NucleotideMatrix*)matrices)[i].name;",
            "        }",
            "",
            "        if (strcasecmp(name, matrix_name) == 0) {",
            "            return i;",
            "        }",
            "    }",
            "",
            "    return -1;",
            "}",
            "",
            "INLINE void list_matrices_for_seq_type(int seq_type) {",
            "    if (seq_type == SEQ_TYPE_AMINO) {",
            "        for (int i = 0; i < NUM_AMINO_MATRICES; i++) {",
            '            printf("  %s%s", ALL_AMINO_MATRICES[i].name,',
            '                  (i + 1) % 5 == 0 ? "\\n" : (i == NUM_AMINO_MATRICES - 1) ? "\\n" : ", ");',
            "        }",
            "    } else if (seq_type == SEQ_TYPE_NUCLEOTIDE) {",
            "        for (int i = 0; i < NUM_NUCLEOTIDE_MATRICES; i++) {",
            '            printf("  %s%s", ALL_NUCLEOTIDE_MATRICES[i].name,',
            '                  (i + 1) % 5 == 0 ? "\\n" : (i == NUM_NUCLEOTIDE_MATRICES - 1) ? "\\n" : ", ");',
            "        }",
            "    }",
            "}",
            "",
        ]
    )

    header_content.append("#endif // MATRICES_H")
    return "\n".join(header_content)


def create_matrices_header():
    header_content = generate_matrix_tables()

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(header_content)

    logger.info(f"Generated {OUTPUT_FILE}")

    for type_name, type_info in MATRIX_TYPES.items():
        if type_info["matrices"]:
            matrix_names = [name for name, _ in type_info["matrices"]]
            logger.info(
                f"{len(matrix_names)} {type_name} matrices: {', '.join(matrix_names)}"
            )


if __name__ == "__main__":
    create_matrices_header()
