#!/bin/bash

# Inserts a newline after closing braces (`}`) in the given file,
# if the next line does not begin with another closing brace.
#
# Usage:
#   ./format.sh <file_path>
#
# Requirements:
#   - Unix-like environment with `sed`
#
# Example:
#   ./format.sh myfile.c
#
# VS Code (Run on Save)
# Example `settings.json`:
# "emeraldwalk.runonsave": {
#     "commands": [
#         {
#             "match": "\\.(?:c|h|cpp|hpp)$",
#             "cmd": "bash /path/to/scripts/format.sh \"${file}\""
#         }
#     ]
# },

if [ $# -lt 1 ]; then
    echo "Usage: format.sh <file_path>"
    exit 1
fi

file_path="$1"

if [ ! -f "$file_path" ]; then
    echo "Error: File $file_path does not exist"
    exit 1
fi

temp_file=$(mktemp)

# Adds newlines after closing braces when needed:
# 1. Find lines ending with '}'
# 2. Pull in next line
# 3. Check different conditions and format accordingly
sed -e '
# Target lines ending with a closing brace
/}$/{
  # Append next line to pattern space
  N
  
  # CASE 1: If next line starts with "}" (with optional whitespace), 
  # do nothing (branch to end)
  /}\n[[:space:]]*}/b
  
  # CASE 2: If next line starts with non-whitespace character,
  # insert a newline after the closing brace
  /}\n[^[:space:]]/ {
    s/}/}\n/
  }
  
  # CASE 3: If next line starts with whitespace followed by 
  # a non-whitespace and non-closing-brace character,
  # insert a newline after the closing brace
  /}\n[[:space:]]\+[^[:space:]}]/ {
    s/}/}\n/
  }
  
  # Print up to first newline and delete it from pattern space
  P
  D
}' "$file_path" > "$temp_file"

mv "$temp_file" "$file_path"

exit 0