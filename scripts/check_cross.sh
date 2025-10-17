#!/bin/bash

MINGW_PKG_CONFIG="x86_64-w64-mingw32-pkg-config"

if ! which $MINGW_PKG_CONFIG >/dev/null 2>&1; then
    echo "Error: $MINGW_PKG_CONFIG not found. Please install mingw-w64-pkg-config."
    exit 1
fi

if ! $MINGW_PKG_CONFIG --exists hdf5; then
    echo "Error: MinGW HDF5 not found. Please install mingw-w64-hdf5."
    exit 1
fi

MINGW_PREFIX=$($MINGW_PKG_CONFIG --variable=prefix hdf5 2>/dev/null || echo /usr/x86_64-w64-mingw32)
DLL_FOLDER="$MINGW_PREFIX/bin"

mkdir -p bin 2>/dev/null

DLLS=(
    "libhdf5.dll"
    "libssp-0.dll"
    "libwinpthread-1.dll"
    "libsz.dll"
    "zlib1.dll"
)

for dll in "${DLLS[@]}"; do
    found=0
    for path in "$DLL_FOLDER" "$MINGW_PREFIX/lib"; do
        if [ -f "$path/$dll" ]; then
            cp -f "$path/$dll" bin/ 2>/dev/null
            found=1
            break
        fi
    done
    if [ $found -eq 0 ]; then
        echo "Warning: $dll not found in $DLL_FOLDER or $MINGW_PREFIX/lib"
    fi
done
