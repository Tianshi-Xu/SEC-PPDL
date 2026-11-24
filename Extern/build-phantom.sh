#!/usr/bin/env bash
set -euo pipefail
extern_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="$extern_dir/phantom-fhe"
revision="$(cat "$extern_dir/phantom-fhe.rev")"
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --filter=blob:none https://github.com/encryptorion-lab/phantom-fhe.git "$source_dir"
    git -C "$source_dir" checkout --detach "$revision"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
    echo "Phantom revision differs from Extern/phantom-fhe.rev; review the checkout before building." >&2
    exit 1
fi
if git -C "$source_dir" apply --check "$extern_dir/patch/phantom.patch" 2>/dev/null; then
    git -C "$source_dir" apply "$extern_dir/patch/phantom.patch"
elif ! git -C "$source_dir" apply --reverse --check "$extern_dir/patch/phantom.patch" 2>/dev/null; then
    echo "Existing Phantom changes do not match the required patch." >&2
    exit 1
fi
cmake -S "$source_dir" -B "$source_dir/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-native}" \
    -DCMAKE_INSTALL_PREFIX="$source_dir/build_phantom" \
    -DPHANTOM_ENABLE_EXAMPLE=OFF -DRNS_POLY_BATCH=ON -DFP64_MM_ARITH=OFF
cmake --build "$source_dir/build" --parallel "${BUILD_JOBS:-8}"
cmake --install "$source_dir/build"
