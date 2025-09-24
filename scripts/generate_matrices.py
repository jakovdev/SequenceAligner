#!/usr/bin/env python3
import parasail
import os
import sys
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

if os.path.exists("scripts/generate_matrices.py"):
    HEADER_FILE = "code/include/core/biology/score/matrices.h"
    SOURCE_FILE = "code/src/core/biology/score/matrices.c"
else:
    HEADER_FILE = "../code/include/core/biology/score/matrices.h"
    SOURCE_FILE = "../code/src/core/biology/score/matrices.c"

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYVBZX*"
NUCLEOTIDES = "ATGCSWRYKMBVHDN*"

MATRIX_TYPES = {
    "amino": {
        "alphabet": AMINO_ACIDS,
        "matrices": [],
        "pattern": r"(blosum\d+|pam\d+)",
    },
    "nucleotide": {
        "alphabet": NUCLEOTIDES,
        "matrices": [],
        "pattern": r"(dnafull|nuc44)",
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


def generate_header_file(matrix_types):
    header_content = [
        "#pragma once",
        "#ifndef MATRICES_H",
        "#define MATRICES_H",
        "// clang-format off",
        "",
    ]

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet_name = type_name.upper()
        header_content.append(
            f"#define {alphabet_name}_SIZE {len(type_info['alphabet'])}"
        )
        header_content.append(f"extern const char {alphabet_name}_ALPHABET[];")
        header_content.append("")

    max_dim = 0
    max_dim_name = ""
    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        dim_size = len(type_info["alphabet"])
        if dim_size > max_dim:
            max_dim = dim_size
            max_dim_name = f"{type_name.upper()}_SIZE"

    header_content.append(f"#define MAX_MATRIX_DIM {max_dim_name}")
    header_content.append("")

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet_name = type_name.upper()
        header_content.append(
            f"#define NUM_{alphabet_name}_MATRICES {len(type_info['matrices'])}"
        )
    header_content.append("")

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet_name = type_name.upper()
        header_content.extend(
            [
                f"typedef struct",
                "{",
                "    const char* name;",
                f"    const int (*matrix)[{alphabet_name}_SIZE];",
                f"}} {type_name.capitalize()}Matrix;",
                "",
            ]
        )

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet_name = type_name.upper()
        header_content.append(
            f"extern const {type_name.capitalize()}Matrix ALL_{alphabet_name}_MATRICES[NUM_{alphabet_name}_MATRICES];"
        )

    header_content.append("")
    header_content.append("// clang-format on")
    header_content.append("#endif // MATRICES_H")
    return "\n".join(header_content)


def generate_source_file(matrix_types):
    source_content = [
        "// clang-format off",
        '#include "core/biology/score/matrices.h"',
        "",
    ]

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet = type_info["alphabet"]
        alphabet_name = type_name.upper()
        source_content.extend(
            [
                f'const char {alphabet_name}_ALPHABET[] = "{alphabet}";',
                "",
            ]
        )

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        source_content.append(f"// {type_name.capitalize()} matrix identifiers")
        source_content.append(f"typedef enum")
        source_content.append("{")
        for i, (name, _) in enumerate(type_info["matrices"]):
            source_content.append(f"    {name}_ID = {i},")
        source_content.append(f"}} {type_name.capitalize()}MatrixID;")
        source_content.append("")

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet_name = type_name.upper()
        source_content.append(f"// {type_name.capitalize()} matrices")

        for name, matrix_obj in type_info["matrices"]:
            source_content.append(
                f"static const int {name}[{alphabet_name}_SIZE][{alphabet_name}_SIZE] = {format_matrix_as_c_array(matrix_obj)};"
            )
            source_content.append("")

    for type_name, type_info in matrix_types.items():
        if not type_info["matrices"]:
            continue

        alphabet_name = type_name.upper()
        source_content.append(
            f"const {type_name.capitalize()}Matrix ALL_{alphabet_name}_MATRICES[NUM_{alphabet_name}_MATRICES] = {{"
        )
        for name, _ in type_info["matrices"]:
            source_content.append(f'    {{"{name}", {name}}},')
        source_content.append("};")
        source_content.append("")

    source_content.append("// clang-format on")
    return "\n".join(source_content)


def create_matrices_files():
    matrix_types = get_available_matrices()

    header_content = generate_header_file(matrix_types)
    source_content = generate_source_file(matrix_types)

    os.makedirs(os.path.dirname(os.path.abspath(HEADER_FILE)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(SOURCE_FILE)), exist_ok=True)

    with open(HEADER_FILE, "w") as f:
        f.write(header_content)

    with open(SOURCE_FILE, "w") as f:
        f.write(source_content)

    logger.info(f"Generated {HEADER_FILE} and {SOURCE_FILE}")

    for type_name, type_info in MATRIX_TYPES.items():
        if type_info["matrices"]:
            matrix_names = [name for name, _ in type_info["matrices"]]
            logger.info(
                f"{len(matrix_names)} {type_name} matrices: {', '.join(matrix_names)}"
            )


if __name__ == "__main__":
    create_matrices_files()
