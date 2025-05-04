# Suppress make recursive messages
MAKEFLAGS += --no-print-directory

# Windows specific settings
IS_WINDOWS := $(if $(filter Windows_NT,$(OS)),yes,)
MAKE_CMD := $(if $(IS_WINDOWS), mingw32-make, make)

# TODO: Check if running MSYS2 from windows terminal and replace RM and MKDIR with:
#RM := $(if $(IS_WINDOWS),powershell -Command "Remove-Item -Force -ErrorAction SilentlyContinue",rm -f)
#$(if $(IS_WINDOWS),powershell -Command "if (-not (Test-Path $(FILE))) { New-Item -ItemType Directory -Path $(FILE) | Out-Null }",mkdir -p $(FILE))
# or move it to a simple .sh script
RM := rm -f
MKDIR := mkdir -p

# Project directories
BIN_DIR := bin
META_DIR := .meta
RESULTS_DIR := results
SCRIPTS_DIR := scripts

# Tracking files
DLL_COMPLETE := $(META_DIR)/dll_complete
SETUP_COMPLETE := $(META_DIR)/setup_complete

# GCC directories
SRC_DIR := src
INCLUDE_DIR := include
OBJ_DIR := $(BIN_DIR)/obj

# GCC files
MAIN_SRC := $(SRC_DIR)/main.c
HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)
SRCS := $(filter-out $(MAIN_SRC),$(wildcard $(SRC_DIR)/*.c))
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
# GCC files for various builds
OBJS_DEBUG := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-debug.o,$(SRCS))
OBJS_X86_64 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-x86-64.o,$(SRCS))
OBJS_AVX2 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-avx2.o,$(SRCS))
OBJS_AVX512 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-avx512.o,$(SRCS))

# GCC flags
HDF5_CFLAGS := $(if $(IS_WINDOWS),-I/ucrt64/include,$(shell bash $(SCRIPTS_DIR)/check_hdf5.sh --parseable | head -n 1))
BASE_CFLAGS := -pthread -I$(INCLUDE_DIR) $(HDF5_CFLAGS) 
RELEASE_CFLAGS := $(BASE_CFLAGS) -O3 -ffast-math -flto -fomit-frame-pointer "-Wl,--gc-sections" -DNDEBUG
DEBUG_CFLAGS := $(BASE_CFLAGS) -g -O0 -Wall -Wextra -Wpedantic -Werror -fstack-protector-strong
# GCC libraries
HDF5_CLIBS := $(if $(IS_WINDOWS),-L/ucrt64/lib -lhdf5 -lz,$(shell bash $(SCRIPTS_DIR)/check_hdf5.sh --parseable | tail -n 1))
CLIBS := -lm $(HDF5_CLIBS) $(if $(IS_WINDOWS),-lShlwapi,)

# GCC build results
CC := gcc
BIN_EXT := $(if $(IS_WINDOWS),.exe,)
MAIN_BIN := $(BIN_DIR)/seqalign$(BIN_EXT)
DEBUG_BIN := $(BIN_DIR)/seqalign-debug$(BIN_EXT)
X86_64_BIN := $(BIN_DIR)/seqalign-x86-64$(BIN_EXT)
AVX2_BIN := $(BIN_DIR)/seqalign-avx2$(BIN_EXT)
AVX512_BIN := $(BIN_DIR)/seqalign-avx512$(BIN_EXT)

.PHONY: all debug clean help setup update _x86-64 _avx2 _avx512

all: CFLAGS := -march=native $(RELEASE_CFLAGS)
all: check-setup $(BIN_DIR) $(OBJ_DIR) $(RESULTS_DIR) $(MAIN_BIN) $(if $(IS_WINDOWS),check-libraries,)

debug: CFLAGS := -march=native $(DEBUG_CFLAGS)
debug: check-setup $(BIN_DIR) $(OBJ_DIR) $(RESULTS_DIR) $(DEBUG_BIN) $(if $(IS_WINDOWS),check-libraries,)

_x86-64: CFLAGS := -march=x86-64 $(RELEASE_CFLAGS)
_x86-64: check-setup $(BIN_DIR) $(OBJ_DIR) $(RESULTS_DIR) $(X86_64_BIN) $(if $(IS_WINDOWS),check-libraries,)

_avx2: CFLAGS := -march=x86-64 -mavx2 $(RELEASE_CFLAGS)
_avx2: check-setup $(BIN_DIR) $(OBJ_DIR) $(RESULTS_DIR) $(AVX2_BIN) $(if $(IS_WINDOWS),check-libraries,)

_avx512: CFLAGS := -march=x86-64 -mavx512f -mavx512bw $(RELEASE_CFLAGS)
_avx512: check-setup $(BIN_DIR) $(OBJ_DIR) $(RESULTS_DIR) $(AVX512_BIN) $(if $(IS_WINDOWS),check-libraries,)

# Directories
$(BIN_DIR):
	@$(MKDIR) $(BIN_DIR)

$(OBJ_DIR):
	@$(MKDIR) $(OBJ_DIR)

$(RESULTS_DIR):
	@$(MKDIR) $(RESULTS_DIR)

$(META_DIR):
	@$(MKDIR) $(META_DIR)

# GCC object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-debug.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@echo "Compiling debug $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-x86-64.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@echo "Compiling x86-64 $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-avx2.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@echo "Compiling AVX2 $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-avx512.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@echo "Compiling AVX512 $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

# GCC build results
$(MAIN_BIN): $(MAIN_SRC) $(OBJS) $(HEADERS) | check-setup
	@echo "Compiling Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(CLIBS) && echo "Build complete! Run the program with: $@"

$(DEBUG_BIN): $(MAIN_SRC) $(OBJS_DEBUG) $(HEADERS) | check-setup
	@echo "Compiling Debug Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS_DEBUG) -o $@ $(CLIBS) && echo "Debug build complete! Run the program with: $@"

$(X86_64_BIN): $(MAIN_SRC) $(OBJS_X86_64) $(HEADERS) | check-setup
	@echo "Compiling x86-64 Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS_X86_64) -o $@ $(CLIBS) && echo "x86-64 build complete!"

$(AVX2_BIN): $(MAIN_SRC) $(OBJS_AVX2) $(HEADERS) | check-setup
	@echo "Compiling AVX2 Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS_AVX2) -o $@ $(CLIBS) && echo "AVX2 build complete!"

$(AVX512_BIN): $(MAIN_SRC) $(OBJS_AVX512) $(HEADERS) | check-setup
	@echo "Compiling AVX512 Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS_AVX512) -o $@ $(CLIBS) && echo "AVX512 build complete!"

clean:
	@echo "Cleaning previous build..."
	@$(RM) $(MAIN_BIN)
	@$(RM) $(DEBUG_BIN)
	@$(RM) $(X86_64_BIN)
	@$(RM) $(AVX2_BIN)
	@$(RM) $(AVX512_BIN)
	@$(RM) -r $(OBJ_DIR)
	@$(if $(IS_WINDOWS),, $(RM) $(patsubst %,%.exe,$(MAIN_BIN)))

$(SETUP_COMPLETE): | $(META_DIR)
	@echo "Checking system configuration..."
ifeq ($(IS_WINDOWS),yes)
	@if [ -z "$$MSYSTEM" ] || [ "$$MSYSTEM" != "UCRT64" ]; then \
        echo "Error: Please run this from the UCRT64 MSYS2 shell"; \
        exit 1; \
    fi
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
	@echo "==============================================="
	@echo "                SequenceAligner                "
	@echo "==============================================="
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