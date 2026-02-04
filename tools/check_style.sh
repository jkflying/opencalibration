#!/bin/bash

DEFAULT_DIRS="include src test app"

# Use provided directories or defaults
DIRS="${@:-$DEFAULT_DIRS}"

# Find all source files
FILES=""
for dir in $DIRS; do
    if [ -d "$dir" ]; then
        FILES="$FILES $(find $dir -iname '*.h' -o -iname '*.cpp' -o -iname '*.hpp')"
    elif [ -f "$dir" ]; then
        FILES="$FILES $dir"
    fi
done

# Check each file with clang-format dry-run
NEEDS_FORMAT=0
for file in $FILES; do
    if [ -n "$file" ]; then
        # --dry-run -Werror returns non-zero if the file would be changed
        if ! clang-format-14 -style='{BasedOnStyle: Microsoft}' --dry-run -Werror "$file" 2>/dev/null; then
            echo "Needs formatting: $file"
            NEEDS_FORMAT=1
        fi
    fi
done

if [ $NEEDS_FORMAT -eq 1 ]; then
    echo ""
    echo "Code style check failed, please run clang-format (e.g. with tools/fix_style.sh)"
    exit 1
fi

echo "Code style check passed"
