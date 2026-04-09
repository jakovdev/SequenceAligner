#!/usr/bin/env bash
set -euo pipefail

BASE_ARCHS=("x86-64" "x86-64-v2" "x86-64-v3" "x86-64-v4")
BUILD_DIR="${PWD}/build"
EXTRA_CMAKE_ARGS=()
mkdir -p "${BUILD_DIR}"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ "${1:-}" == "cross" || "${1:-}" == "mingw" ]]; then
        if command -v x86_64-w64-mingw32-gcc &> /dev/null; then
            EXTRA_CMAKE_ARGS+=("--toolchain ${PWD}/script/toolchain-mingw.cmake")
        else
            echo ""
            echo "MinGW cross-compiler not detected. Will not build anything."
            exit 1
        fi
    fi
elif [ "$MSYSTEM" == "UCRT64" ]; then
    if ! command -v gcc &> /dev/null; then
        echo ""
        echo "MinGW UCRT64 compiler not detected. Will not build anything."
        echo "Please install the following packages:"
        echo ""
        echo "pacman -Syu"
        echo "pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-hdf5"
        exit 1
    fi
else
    echo "Unknown platform: $OSTYPE"
    exit 1
fi

for arch in "${BASE_ARCHS[@]}"; do
    echo ""

    build_subdir="${BUILD_DIR}/${arch}"
    rm -rf "${build_subdir}"
    mkdir -p "${build_subdir}"

    cmake -S . -B "${build_subdir}" -G Ninja -DARCH_LEVEL="${arch}" "${EXTRA_CMAKE_ARGS[@]}"
    cmake --build "${build_subdir}" -- -j $(nproc)
    (cd "${build_subdir}" && cpack)
done

echo ""
echo "Done."
