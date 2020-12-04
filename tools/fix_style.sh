#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <src_file | dir>"
    echo ""
    echo "ERROR: At least one source file or one directory must be provided!"

    exit 1
fi

for arg in "$@"
do
    if [ -f $arg ]; then
        clang-format-10 -i -style='{BasedOnStyle: Microsoft}' $arg
    elif [ -d $arg ]; then
        find $arg -iname '*.h' -o -iname '*.cpp' -o -iname '*.hpp' | xargs clang-format-10 -i -style='{BasedOnStyle: Microsoft}'
        find $arg -iname '*.h' -o -iname '*.cpp' -o -iname '*.hpp' | xargs chmod 644
    fi
done
