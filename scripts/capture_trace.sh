#!/bin/sh
set -eu
if [ "$#" -lt 3 ]; then
    echo "usage: $0 <tool:valgrind|pin|dynamoRIO> <program> <output.trace> [program args...]" >&2
    exit 2
fi
tool=$1
program=$2
output=$3
shift 3
case "$tool" in
  valgrind)
    command -v valgrind >/dev/null 2>&1 || { echo "Valgrind is not installed; no trace generated." >&2; exit 3; }
    echo "Use Valgrind Lackey with --trace-mem=yes and convert data records to $output." >&2
    exit 4
    ;;
  pin)
    command -v pin >/dev/null 2>&1 || { echo "Intel Pin is not installed; no trace generated." >&2; exit 3; }
    echo "Provide a Pin load-trace pintool and write one hexadecimal data address per line to $output." >&2
    exit 4
    ;;
  dynamoRIO)
    command -v drrun >/dev/null 2>&1 || { echo "DynamoRIO is not installed; no trace generated." >&2; exit 3; }
    echo "Provide a DynamoRIO drmemtrace configuration and convert data records to $output." >&2
    exit 4
    ;;
  *)
    echo "unsupported tool: $tool" >&2
    exit 2
    ;;
esac
