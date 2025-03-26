IS_WINDOWS := $(if $(filter Windows_NT,$(OS)),yes,)
# TODO: Check if running MSYS2 from windows terminal and replace RM and MKDIR with:
#RM := $(if $(IS_WINDOWS),powershell -Command "Remove-Item -Force -ErrorAction SilentlyContinue",rm -f)
#$(if $(IS_WINDOWS),powershell -Command "if (-not (Test-Path $(FILE))) { New-Item -ItemType Directory -Path $(FILE) | Out-Null }",mkdir -p $(FILE))
RM := rm -f
MKDIR := mkdir -p
MAKE_CMD := $(if $(IS_WINDOWS), mingw32-make, make)

# Project directories
SRC_DIR := src
INCLUDE_DIR := include
BIN_DIR := bin
RESULTS_DIR := results
SCRIPTS_DIR := scripts
META_DIR := .meta

# Build configuration
CC := gcc
BIN_EXT := $(if $(IS_WINDOWS),.exe,)
MAIN_SRC := $(SRC_DIR)/main.c
MAIN_BIN := $(BIN_DIR)/seqalign$(BIN_EXT)
HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)

# Tracking files
SETUP_COMPLETE := $(META_DIR)/setup_complete
DLL_COMPLETE := $(META_DIR)/libraries_complete

# Suppress make recursive messages
MAKEFLAGS += --no-print-directory

# HDF5 configuration
ifeq ($(IS_WINDOWS),yes)
    HDF5_CFLAGS := -I/ucrt64/include
    HDF5_LIBS := -L/ucrt64/lib -lhdf5 -lz
else
    HDF5_CFLAGS := $(shell bash $(SCRIPTS_DIR)/check_hdf5.sh --parseable | head -n 1)
    HDF5_LIBS := $(shell bash $(SCRIPTS_DIR)/check_hdf5.sh --parseable | tail -n 1)
endif

# Compiler flags
BASE_FLAGS := -march=native -pthread -I$(INCLUDE_DIR) $(HDF5_CFLAGS)
OPT_FLAGS := -O3 -ffast-math -funroll-loops -fno-strict-aliasing -flto \
             -fprefetch-loop-arrays "-Wl,--gc-sections" -DNDEBUG
DBG_FLAGS := -g -O0 -Wall -Wextra -Wpedantic -Werror -fstack-protector-strong

CFLAGS := $(BASE_FLAGS) $(if $(filter debug,$(MAKECMDGOALS)),$(DBG_FLAGS),$(OPT_FLAGS))

LIBS := -lm $(HDF5_LIBS) $(if $(IS_WINDOWS),-lShlwapi,)

.PHONY: all debug clean help setup update

all: check-setup $(BIN_DIR) $(RESULTS_DIR) $(MAIN_BIN) $(if $(IS_WINDOWS),check-libraries,)
	@if [ -n "$$BUILD_OCCURRED" ]; then echo "Build complete! Run the program with: $(MAIN_BIN)"; fi

debug: CFLAGS := $(BASE_FLAGS) $(DBG_FLAGS)
debug: check-setup $(BIN_DIR) $(RESULTS_DIR) $(MAIN_BIN) $(if $(IS_WINDOWS),check-libraries,)
	@if [ -n "$$BUILD_OCCURRED" ]; then echo "Debug build complete! Run the program with: $(MAIN_BIN)"; fi

$(BIN_DIR):
	@$(MKDIR) $(BIN_DIR)

$(RESULTS_DIR):
	@$(MKDIR) $(RESULTS_DIR)

$(META_DIR):
	@$(MKDIR) $(META_DIR)

$(MAIN_BIN): $(MAIN_SRC) $(HEADERS) | check-setup
	@echo "Compiling Sequence Aligner..."
	@export BUILD_OCCURRED=true; \
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	@echo "Cleaning previous build..."
	@$(RM) $(MAIN_BIN)
	@$(if $(IS_WINDOWS),, $(RM) $(patsubst %,%.exe,$(MAIN_BIN)))

$(SETUP_COMPLETE): | $(META_DIR)
	@echo "Checking system configuration..."
ifeq ($(IS_WINDOWS),yes)
	@if [ ! -f "/ucrt64/include/hdf5.h" ]; then \
        echo "Required tools not found. Installing..."; \
        bash $(SCRIPTS_DIR)/msys2_setup.sh; \
    fi
else
	@which $(CC) >/dev/null 2>&1 || (echo "Error: Compiler not found. Please install GCC." && exit 1)
	@bash $(SCRIPTS_DIR)/check_hdf5.sh --quiet || echo "Warning: Required packages not found. Build might fail."
endif
	@touch $@

check-setup: $(SETUP_COMPLETE)

setup:
	@echo "Setting up required tools..."
	@$(RM) $(SETUP_COMPLETE)
	@$(MAKE) check-setup

$(DLL_COMPLETE): | $(META_DIR) $(BIN_DIR)
ifeq ($(IS_WINDOWS),yes)
	@bash $(SCRIPTS_DIR)/copy_dlls.sh
	@touch $@
endif

check-libraries: $(DLL_COMPLETE)

update:
	@echo "Updating required tools..."
ifeq ($(IS_WINDOWS),yes)
	@$(RM) $(DLL_COMPLETE)
	@$(MAKE) check-libraries
else
	@echo "Nothing to update on this platform."
endif

help:
	@echo "SequenceAligner: DNA/RNA Sequence Analysis Tool"
	@echo "=============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  $(MAKE_CMD)              - Build the program"
	@echo "  $(MAKE_CMD) debug        - Build with debugging information"
	@echo "  $(MAKE_CMD) clean        - Remove built files"
	@echo "  $(MAKE_CMD) setup        - Install required tools"
	@echo "  $(MAKE_CMD) update       - Update required tools"
	@echo "  $(MAKE_CMD) help         - Show this help message"
	@echo ""
	@echo "After building, run the program with: $(MAIN_BIN)"