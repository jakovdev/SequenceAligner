#!/bin/bash

if [[ "$(uname -o)" != "Msys" ]]; then
    echo "Error: This must be run from the MSYS2 environment"
    exit 1
fi

mkdir -p bin results .meta 2>/dev/null

echo "Installing required tools..."
pacman -S --needed --noconfirm \
    mingw-w64-ucrt-x86_64-gcc \
    mingw-w64-ucrt-x86_64-make \
    mingw-w64-ucrt-x86_64-hdf5

if [ $? -ne 0 ]; then
    echo "Installation failed. Please try running 'pacman -Syu' first to update your system."
    exit 1
fi

echo "Installation complete!"
echo "You can now build the program with 'mingw32-make'."