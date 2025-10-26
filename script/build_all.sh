#!/usr/bin/env bash
set -euo pipefail

BASE_ARCHS=("x86-64" "x86-64-v2" "x86-64-v3" "x86-64-v4")
BUILD_DIR="${PWD}/build"
mkdir -p "${BUILD_DIR}"

CUDA_AVAILABLE="no"
if command -v nvcc &> /dev/null; then
    CUDA_AVAILABLE="yes"
fi

PRESETS=()
for a in "${BASE_ARCHS[@]}"; do
    PRESETS+=("${a}")
    if [ "$CUDA_AVAILABLE" == "yes" ]; then
        PRESETS+=("${a}-cuda")
    fi
done

for preset in "${PRESETS[@]}"; do
    echo ""
    USE_CUDA="OFF"
    if [[ "${preset}" == *"-cuda" ]]; then
        USE_CUDA="ON"
        arch="${preset%-cuda}"
    else
        arch="${preset}"
    fi

    build_subdir="${BUILD_DIR}/${preset}"
    rm -rf "${build_subdir}"
    mkdir -p "${build_subdir}"

    cmake -S . -B "${build_subdir}" -G Ninja -DARCH_LEVEL="${arch}" -DUSE_CUDA="${USE_CUDA}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_subdir}" --config Release -- -j $(nproc)
    (cd "${build_subdir}" && cpack)
done

echo ""
echo "Done."
