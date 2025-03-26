#!/bin/bash

echo "Setting up required program files..."

UCRT64_BIN="/ucrt64/bin"

if [ ! -d "$UCRT64_BIN" ]; then
    echo "Error: Required files not found at $UCRT64_BIN"
    echo "Please make sure you're running from the MSYS2 UCRT64 environment"
    exit 1
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

echo "Setup complete."