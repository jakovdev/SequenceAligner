#!/bin/bash

# Arguments
MODE="check"
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --parseable)
            MODE="parseable"
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

check_mode() {
    $QUIET || echo "Checking required tools..."

    if command -v pkg-config >/dev/null 2>&1; then
        if pkg-config --exists hdf5; then
            $QUIET || echo "All required tools found"
            exit 0
        else
            $QUIET || echo "Warning: Required tools not found. Using fallback settings."
        fi
    else
        $QUIET || echo "Warning: Package configuration tool not found. Using fallback settings."
    fi

    # Check common locations
    if [ -f /usr/include/hdf5.h ]; then
        $QUIET || echo "Required tools found in standard system location"
        exit 0
    elif [ -f /usr/include/hdf5/serial/hdf5.h ]; then
        $QUIET || echo "Required tools found in Ubuntu/Debian path"
        exit 0
    else
        $QUIET || echo "Warning: Required tools not found in common locations. Build may fail."
        $QUIET || echo "Please install the required packages:"
        $QUIET || echo "On Ubuntu/Debian: sudo apt-get install libhdf5-dev"
        $QUIET || echo "On Fedora/RHEL: sudo dnf install hdf5-devel"
        $QUIET || echo "On Arch Linux: sudo pacman -S hdf5"
        exit 1
    fi
}

parseable_mode() {
    if [ -n "$GITHUB_ACTIONS" ]; then
        echo "-I/usr/include/hdf5/serial"
        echo "-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"
        exit 0
    fi

    if command -v pkg-config >/dev/null 2>&1; then
        if pkg-config --exists hdf5; then
            # First line: CFLAGS, Second line: LIBS
            echo "$(pkg-config --cflags hdf5)"
            echo "$(pkg-config --libs hdf5)"
            exit 0
        fi
    fi

    if [ -f /usr/include/hdf5.h ]; then
        # Standard system location (empty CFLAGS)
        echo ""
        echo "-lhdf5"
        exit 0
    elif [ -f /usr/include/hdf5/serial/hdf5.h ]; then
        # Ubuntu/Debian path
        echo "-I/usr/include/hdf5/serial"
        echo "-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"
        exit 0
    else
        # Last resort fallback
        echo ""
        echo "-lhdf5"
        exit 1
    fi
}

if [ "$MODE" = "parseable" ]; then
    parseable_mode
else
    check_mode
fi
