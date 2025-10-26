#!/usr/bin/env python3
# Inserts a newline after closing braces (`}`) in the given file,
# if the next line does not begin with another closing brace.
#
# Usage:
#   python3 format.py <file_path>
#
# Requirements:
#   - Python 3
#
# Example:
#   python3 format.py myfile.c
#
# VS Code (Run on Save)
# Example `settings.json`:
# "emeraldwalk.runonsave": {
#     "commands": [
#         {
#             "match": "\\.(?:c|h|cpp|hpp)$",
#             "cmd": "python3 /path/to/script/format.py \"${file}\""
#         }
#     ]
# },

import os
import sys


def add_newlines_after_braces(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        new_lines.append(lines[i])

        if lines[i].rstrip().endswith("}"):
            if (
                i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].strip().startswith("}")
            ):
                new_lines.append("\n")

        i += 1

    with open(file_path, "w") as f:
        f.writelines(new_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: format.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    add_newlines_after_braces(file_path)


if __name__ == "__main__":
    main()
