#!/usr/bin/env bash
set -euo pipefail

BASE_ARCHS=("x86-64" "x86-64-v2" "x86-64-v3" "x86-64-v4")
BUILD_DIR="${PWD}/build"
mkdir -p "${BUILD_DIR}"

PRESETS=()
MINGW_AVAILABLE="no"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v x86_64-w64-mingw32-gcc &> /dev/null; then
        MINGW_AVAILABLE="yes"
        for a in "${BASE_ARCHS[@]}"; do
            PRESETS+=("${a}-mingw")
        done
    else
        echo ""
        echo "MinGW cross-compiler not detected. Will not build anything."
        exit 1
    fi
elif [ "$MSYSTEM" == "UCRT64" ]; then
    if command -v gcc &> /dev/null; then
        MINGW_AVAILABLE="yes"
        for a in "${BASE_ARCHS[@]}"; do
            PRESETS+=("${a}-msys2")
        done
    else
        echo ""
        echo "MinGW UCRT64 compiler not detected. Will not build anything."
        echo "Please install the following packages:"
        echo ""
        echo "pacman -Syu"
        echo "pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-hdf5"
        exit 1
    fi
else
    echo "Unknown platform: $OSTYPE"
    exit 1
fi

for preset in "${PRESETS[@]}"; do
    echo ""
    MINGW_FLAG=""
    if [[ "${preset}" == *"-mingw" ]]; then
        MINGW_FLAG="-DUSE_MINGW=ON"
        arch="${preset%-mingw}"
    else
        arch="${preset%-msys2}"
    fi

    build_subdir="${BUILD_DIR}/${preset}"
    rm -rf "${build_subdir}"
    mkdir -p "${build_subdir}"

    cmake -S . -B "${build_subdir}" -G Ninja -DARCH_LEVEL="${arch}" -DCMAKE_BUILD_TYPE=Release ${MINGW_FLAG}
    cmake --build "${build_subdir}" -- -j $(nproc)
    (cd "${build_subdir}" && cpack)
done

echo ""
echo "Done."
