#!/bin/bash

DEFAULT_DIRS="include src test app"

# Use provided directories or defaults
DIRS="${@:-$DEFAULT_DIRS}"

for arg in $DIRS
do
    if [ -f $arg ]; then
        clang-format-14 -i -style='{BasedOnStyle: Microsoft}' $arg
    elif [ -d $arg ]; then
        find $arg -iname '*.h' -o -iname '*.cpp' -o -iname '*.hpp' | xargs clang-format-14 -i -style='{BasedOnStyle: Microsoft}'
        find $arg -iname '*.h' -o -iname '*.cpp' -o -iname '*.hpp' | xargs chmod 644
    fi
done
