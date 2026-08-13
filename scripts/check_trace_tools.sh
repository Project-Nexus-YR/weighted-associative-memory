#!/bin/sh
set -u
for tool in valgrind pin drrun perf; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "$tool=available"
    else
        echo "$tool=unavailable"
    fi
done
echo "clang=$(command -v clang 2>/dev/null || echo unavailable)"
echo "gcc=$(command -v gcc 2>/dev/null || echo unavailable)"
