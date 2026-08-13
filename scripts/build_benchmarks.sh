#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$root/build/benchmarks"
for source in "$root"/benchmarks/*.c; do
    name=$(basename "$source" .c)
    cc=${CC:-cc}
    "$cc" -O2 -std=c11 -Wall -Wextra -Wno-unused-function "$source" -o "$root/build/benchmarks/$name"
done
echo "Built benchmarks in $root/build/benchmarks"
