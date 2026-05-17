#!/bin/bash

set -e
set +x

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_folder output_folder"
    exit 1
fi

input_folder="$1"
output_folder="$2"

mkdir -p "$output_folder"

# Get the directory where the current script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer PcapSplitter from the same directory as this script (installed by _install_binaries),
# fall back to PATH for standard installs.
if [ -x "$script_dir/PcapSplitter" ]; then
    pcap_splitter="$script_dir/PcapSplitter"
elif command -v PcapSplitter &>/dev/null; then
    pcap_splitter="PcapSplitter"
else
    echo "Error: PcapSplitter not found in $script_dir or PATH." >&2
    exit 2
fi

find "$input_folder" -type f | parallel "mkdir -p $output_folder/{/.} && $pcap_splitter -f {} -o $output_folder/{/.}/ -m connection"
