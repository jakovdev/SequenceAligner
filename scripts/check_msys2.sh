#!/bin/bash

if [ -z "$MSYSTEM" ] || [ "$MSYSTEM" != "UCRT64" ]; then
    echo "Error: Please run this from the UCRT64 MSYS2 shell"
    exit 1
fi

UCRT64_BIN="/ucrt64/bin"

if [ ! -d "$UCRT64_BIN" ]; then
    echo "Error: Required files not found at $UCRT64_BIN"
    echo "Please make sure you're running from the MSYS2 UCRT64 environment"
    exit 1
fi

if [ ! -f "/ucrt64/include/hdf5.h" ]; then
    echo "Required tools not found. Installing..."
    pacman -S --needed --noconfirm \
        mingw-w64-ucrt-x86_64-gcc \
        mingw-w64-ucrt-x86_64-make \
        mingw-w64-ucrt-x86_64-hdf5
    if [ $? -ne 0 ]; then
        echo "Installation failed. Please try running 'pacman -Syu' first to update your system."
        exit 1
    fi
fi

mkdir -p bin 2>/dev/null

DLLS=(
    "libhdf5-310.dll"
    "libcrypto-3-x64.dll"
    "libwinpthread-1.dll"
    "libcurl-4.dll"
    "libsz-2.dll"
    "zlib1.dll"
    "libbrotlidec.dll"
    "libidn2-0.dll"
    "libnghttp2-14.dll"
    "libnghttp3-9.dll"
    "libpsl-5.dll"
    "libssh2-1.dll"
    "libbrotlicommon.dll"
    "libzstd.dll"
    "libssl-3-x64.dll"
    "libintl-8.dll"
    "libiconv-2.dll"
    "libunistring-5.dll"
)

for dll in "${DLLS[@]}"; do
    cp -f "$UCRT64_BIN/$dll" bin/ 2>/dev/null
done
