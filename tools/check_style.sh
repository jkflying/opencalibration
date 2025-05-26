#!/bin/bash
set -e

# Fix style recursively in all the repository
bash tools/fix_style.sh include src test app

# Print the diff with the remote branch (empty if no diff)
git --no-pager diff -U0 --color

# Check if there are changes, and failed
if ! git diff-index --quiet HEAD --; then echo "Code style check failed, please run clang-format (e.g. with tools/fix_style.sh)"; exit 1; fi
