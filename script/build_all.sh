#!/usr/bin/env bash
set -euo pipefail

BASE_ARCHS=("x86-64" "x86-64-v2" "x86-64-v3" "x86-64-v4")
BUILD_DIR="${PWD}/build"
CMAKE="cmake"
mkdir -p "${BUILD_DIR}"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ "${1:-}" == "cross" || "${1:-}" == "mingw" ]]; then
        CMAKE="x86_64-w64-mingw32-cmake"
        if ! command -v $CMAKE &> /dev/null; then
            echo ""
            echo "$CMAKE not detected. Will not build anything."
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
        echo "pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-tools mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-hdf5"
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

    $CMAKE -S . -B "${build_subdir}" -G Ninja -DARCH_LEVEL="${arch}"
    cmake --build "${build_subdir}" -- -j $(nproc)
    (cd "${build_subdir}" && cpack)
done

echo ""
echo "Done."
