# Suppress make recursive messages + use all available CPU cores
MAKEFLAGS += --no-print-directory -j$(shell nproc)
.PHONY: native cuda help clean update setup debug cuda-debug all check-setup check-libraries check-cuda _x86-64-v1 _x86-64-v2 _x86-64-v3 _x86-64-v4

# OS specific settings
OS ?= $(shell uname -s)
IS_WINDOWS := $(if $(filter Windows_NT,$(OS)),yes,)
MAKE_CMD := $(if $(IS_WINDOWS), mingw32-make, make)
.DEFAULT_GOAL := native

help:
	@echo "==============================================="
	@echo "                SequenceAligner                "
	@echo "==============================================="
	@echo ""
	@echo "Available commands:"
	@echo "  $(MAKE_CMD) native       - Build the program for your specific PC"
	@echo "  $(MAKE_CMD) clean        - Remove built files"
	@echo "  $(MAKE_CMD) setup        - Check if required tools exist"
	@echo "  $(MAKE_CMD) update       - Update required tools"
	@echo "  $(MAKE_CMD) help         - Show this help message"
ifneq ($(IS_WINDOWS),yes)
	@echo ""
	@echo "CUDA Commands:"
	@echo "  $(MAKE_CMD) cuda         - Build with CUDA support"
endif
	@echo ""
	@echo "After building, run the program with: $(MAIN_BIN)"

# TODO: Check if running MSYS2 from windows terminal and replace RM and MKDIR with:
#RM := $(if $(IS_WINDOWS),powershell -Command "Remove-Item -Force -ErrorAction SilentlyContinue",rm -f)
#$(if $(IS_WINDOWS),powershell -Command "if (-not (Test-Path $(FILE))) { New-Item -ItemType Directory -Path $(FILE) | Out-Null }",mkdir -p $(FILE))
# or move it to a simple .sh script
RM := rm -f
MKDIR := mkdir -p
check_dir_exists = $(if $(wildcard $1),,$(error Error: $2 directory ($1) not found! Please don't move the Makefile without the required directories))

# Project directories
BIN_DIR := bin
META_DIR := .meta
RESULTS_DIR := results
SCRIPTS_DIR := scripts
$(call check_dir_exists,$(SCRIPTS_DIR),Scripts)

# Tracking files
DLL_COMPLETE := $(META_DIR)/dll_complete
CUDA_COMPLETE := $(META_DIR)/cuda_complete
SETUP_COMPLETE := $(META_DIR)/setup_complete

# GCC directories
SRC_DIR := code/src
$(call check_dir_exists,$(SRC_DIR),Source)
INCLUDE_DIR := code/include
$(call check_dir_exists,$(INCLUDE_DIR),Include)
OBJ_DIR := $(BIN_DIR)/obj

# GCC files
MAIN_SRC := $(SRC_DIR)/main.c
HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)
rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))
SRCS := $(filter-out $(MAIN_SRC),$(call rwildcard,$(SRC_DIR)/,*.c))
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
# GCC files for various builds
OBJS_DEBUG := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-debug.o,$(SRCS))
OBJS_X86_64_V1 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-x86-64-v1.o,$(SRCS))
OBJS_X86_64_V2 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-x86-64-v2.o,$(SRCS))
OBJS_X86_64_V3 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-x86-64-v3.o,$(SRCS))
OBJS_X86_64_V4 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-x86-64-v4.o,$(SRCS))

# GCC flags
HDF5_CFLAGS := $(if $(IS_WINDOWS),-I/ucrt64/include,$(shell bash $(SCRIPTS_DIR)/check_hdf5.sh --parseable | head -n 1))
BASE_CFLAGS := -pthread -I$(INCLUDE_DIR) $(HDF5_CFLAGS) 
RELEASE_CFLAGS := $(BASE_CFLAGS) -O3 -ffast-math -flto -fomit-frame-pointer "-Wl,--gc-sections" -DNDEBUG
DEBUG_CFLAGS := $(BASE_CFLAGS) -g -O0 -Wall -Wextra -Werror \
                -Wshadow -Wconversion -Wmissing-declarations \
                -Wundef -Wfloat-equal -Wcast-align \
                -Wstrict-prototypes -Wswitch-enum -pedantic \
                -fstack-protector-strong -fsanitize=address -fanalyzer
# GCC libraries
HDF5_CLIBS := $(if $(IS_WINDOWS),-L/ucrt64/lib -lhdf5 -lz,$(shell bash $(SCRIPTS_DIR)/check_hdf5.sh --parseable | tail -n 1))
CLIBS := $(HDF5_CLIBS) $(if $(IS_WINDOWS),-lShlwapi,)

# GCC build results
CC := gcc
BIN_EXT := $(if $(IS_WINDOWS),.exe,)
MAIN_BIN := $(BIN_DIR)/seqalign$(BIN_EXT)
DEBUG_BIN := $(BIN_DIR)/seqalign-debug$(BIN_EXT)
X86_64_V1_BIN := $(BIN_DIR)/seqalign-x86-64-v1$(BIN_EXT)
X86_64_V2_BIN := $(BIN_DIR)/seqalign-x86-64-v2$(BIN_EXT)
X86_64_V3_BIN := $(BIN_DIR)/seqalign-x86-64-v3$(BIN_EXT)
X86_64_V4_BIN := $(BIN_DIR)/seqalign-x86-64-v4$(BIN_EXT)

# CUDA directories
CUDA_OBJ_DIR := $(BIN_DIR)/cuda_obj
ifneq ($(filter cuda cuda-debug,$(MAKECMDGOALS)),)
CUDA_SRC_DIR := code/cuda/src
$(call check_dir_exists,$(CUDA_SRC_DIR),CUDA source)
CUDA_INCLUDE_DIR := code/cuda/include
$(call check_dir_exists,$(CUDA_INCLUDE_DIR),CUDA include)
CUDA_C_BINDINGS_DIR := code/cuda/c_bindings

# CUDA files
CUDA_SRCS := $(wildcard $(CUDA_SRC_DIR)/*.cu)
CUDA_HEADERS := $(wildcard $(CUDA_INCLUDE_DIR)/*.{h,hpp,cuh}) $(wildcard $(CUDA_C_BINDINGS_DIR)/*.{h,hpp,cuh})
HEADERS += $(wildcard $(CUDA_C_BINDINGS_DIR)/*.{h,hpp,cuh})
CUDA_OBJS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_SRCS))
CUDA_C_OBJS := $(patsubst $(SRC_DIR)/%.c,$(CUDA_OBJ_DIR)/%.o,$(SRCS))
# CUDA files for various builds
CUDA_OBJS_DEBUG := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%-debug.o,$(CUDA_SRCS))
CUDA_C_OBJS_DEBUG := $(patsubst $(SRC_DIR)/%.c,$(CUDA_OBJ_DIR)/%-debug.o,$(SRCS))

# CUDA flags and libraries
CUDA_RELEASE_FLAGS := -std=c++20 -O3 -Xcompiler "-fPIC" -I$(CUDA_INCLUDE_DIR) -I$(CUDA_C_BINDINGS_DIR) -Wno-deprecated-gpu-targets
CUDA_DEBUG_FLAGS := -g -G -O0 -Xcompiler "-Wall -fPIC" -I$(CUDA_INCLUDE_DIR) -I$(CUDA_C_BINDINGS_DIR) -Wno-deprecated-gpu-targets
CUDA_LIBS := $(shell pkg-config --libs cuda) -lcudart
CUDA_CFLAGS := -DUSE_CUDA -I$(CUDA_C_BINDINGS_DIR)

# CUDA build results
NVCC := nvcc
endif
CUDA_BIN := $(BIN_DIR)/seqalign-cuda$(BIN_EXT)
CUDA_DEBUG_BIN := $(BIN_DIR)/seqalign-cuda-debug$(BIN_EXT)

native: CFLAGS := -march=native $(RELEASE_CFLAGS)
native: $(MAIN_BIN)
all: native # Only build native since it's probably what the user wants

debug: CFLAGS := -march=native $(DEBUG_CFLAGS)
debug: $(DEBUG_BIN)

# See: https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels for more details

# baseline for all x86-64 CPUs, Intel Prescott (2004+) and AMD K8 (2003+)
_x86-64-v1: CFLAGS := -march=x86-64 $(RELEASE_CFLAGS)
_x86-64-v1: $(X86_64_V1_BIN)

# Intel Nehalem (2008+) and (Atom) Silvermont (SoC) (2013+), AMD Bulldozer (2011+) and Jaguar (2013+)
_x86-64-v2: CFLAGS := -march=x86-64-v2 $(RELEASE_CFLAGS)
_x86-64-v2: $(X86_64_V2_BIN)

# Intel Haswell (2013+) with AVX2 enabled models and (Atom) Gracemont (SoC) (2021+), AMD Excavator (2015+) and Zen 1+ (2017+)
_x86-64-v3: CFLAGS := -march=x86-64-v3 $(RELEASE_CFLAGS)
_x86-64-v3: $(X86_64_V3_BIN)

# Intel Skylake (2015+) with AVX512 enabled models, AMD Zen 4+ (2022+)
_x86-64-v4: CFLAGS := -march=x86-64-v4 $(RELEASE_CFLAGS)
_x86-64-v4: $(X86_64_V4_BIN)

cuda: CFLAGS := -march=native $(RELEASE_CFLAGS) $(CUDA_CFLAGS)
cuda: CLIBS += $(CUDA_LIBS) -lstdc++
cuda: $(CUDA_BIN)

cuda-debug: CFLAGS := -march=native $(DEBUG_CFLAGS) $(CUDA_CFLAGS) -fno-sanitize=address
cuda-debug: CLIBS += $(CUDA_LIBS) -lstdc++
cuda-debug: $(CUDA_DEBUG_BIN)

# Directories
$(BIN_DIR):
	@$(MKDIR) $(BIN_DIR)

$(OBJ_DIR):
	@$(MKDIR) $(OBJ_DIR)

$(CUDA_OBJ_DIR):
	@$(MKDIR) $(CUDA_OBJ_DIR)

$(RESULTS_DIR):
	@$(MKDIR) $(RESULTS_DIR)

$(META_DIR):
	@$(MKDIR) $(META_DIR)

# GCC object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-debug.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "Compiling debug $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-x86-64-v1.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "Compiling for x86-64-v1 $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-x86-64-v2.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "Compiling for x86-64-v2 $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-x86-64-v3.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "Compiling for x86-64-v3 $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-x86-64-v4.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "Compiling for x86-64-v4 $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

# GCC build results
$(MAIN_BIN): $(MAIN_SRC) $(OBJS) $(HEADERS) | $(BIN_DIR) check-setup check-libraries
	@if [ -L "$(MAIN_BIN)" ]; then \
        echo "Removing existing symlink at $(MAIN_BIN)"; \
        $(RM) $(MAIN_BIN); \
    fi
	@echo "Compiling Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(CLIBS) && echo "Build complete! Run the program with: $@"

$(DEBUG_BIN): $(MAIN_SRC) $(OBJS_DEBUG) $(HEADERS) | $(BIN_DIR) check-setup check-libraries
	@echo "Compiling Debug Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS_DEBUG) -o $@ $(CLIBS) && echo "Debug build complete! Run the program with: $@"

$(X86_64_V1_BIN): $(MAIN_SRC) $(OBJS_X86_64_V1) $(HEADERS) | $(BIN_DIR) check-setup check-libraries
	@echo "Compiling x86-64-v1 Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS_X86_64_V1) -o $@ $(CLIBS) && echo "x86-64-v1 build complete!"

$(X86_64_V2_BIN): $(MAIN_SRC) $(OBJS_X86_64_V2) $(HEADERS) | $(BIN_DIR) check-setup check-libraries
	@echo "Compiling x86-64-v2 Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS_X86_64_V2) -o $@ $(CLIBS) && echo "x86-64-v2 build complete!"

$(X86_64_V3_BIN): $(MAIN_SRC) $(OBJS_X86_64_V3) $(HEADERS) | $(BIN_DIR) check-setup check-libraries
	@echo "Compiling x86-64-v3 Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS_X86_64_V3) -o $@ $(CLIBS) && echo "x86-64-v3 build complete!"

$(X86_64_V4_BIN): $(MAIN_SRC) $(OBJS_X86_64_V4) $(HEADERS) | $(BIN_DIR) check-setup check-libraries
	@echo "Compiling x86-64-v4 Sequence Aligner..."
	@$(CC) $(CFLAGS) $< $(OBJS_X86_64_V4) -o $@ $(CLIBS) && echo "x86-64-v4 build complete!"

# CUDA object files
$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu $(CUDA_HEADERS) | $(CUDA_OBJ_DIR)
	@echo "Compiling CUDA source: $<"
	@$(NVCC) $(CUDA_RELEASE_FLAGS) -c $< -o $@

$(CUDA_OBJ_DIR)/%-debug.o: $(CUDA_SRC_DIR)/%.cu $(CUDA_HEADERS) | $(CUDA_OBJ_DIR)
	@echo "Compiling CUDA debug source: $<"
	@$(NVCC) $(CUDA_DEBUG_FLAGS) -c $< -o $@
# CUDA C object files
$(CUDA_OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(CUDA_OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "Compiling CUDA C source: $<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(CUDA_OBJ_DIR)/%-debug.o: $(SRC_DIR)/%.c $(HEADERS) | $(CUDA_OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "Compiling CUDA C debug source: $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# CUDA build results
$(CUDA_BIN): $(MAIN_SRC) $(HEADERS) $(CUDA_OBJS) $(CUDA_C_OBJS) $(CUDA_HEADERS) | $(BIN_DIR) check-cuda check-libraries
	@echo "Compiling Sequence Aligner with CUDA..."
	@$(CC) $(CFLAGS) $< $(CUDA_OBJS) $(CUDA_C_OBJS) -o $@ $(CLIBS) && echo "CUDA build complete! Run the program with: $@"
	@if [ ! -f "$(MAIN_BIN)" ] && [ -f "$(CUDA_BIN)" ]; then \
        echo "Creating symlink from $(MAIN_BIN) to $(CUDA_BIN)"; \
        ln -sf $(notdir $(CUDA_BIN)) $(MAIN_BIN); \
    fi

$(CUDA_DEBUG_BIN): $(MAIN_SRC) $(HEADERS) $(CUDA_OBJS_DEBUG) $(CUDA_C_OBJS_DEBUG) $(CUDA_HEADERS) | $(BIN_DIR) check-cuda check-libraries
	@echo "Compiling Debug Sequence Aligner with CUDA..."
	@$(CC) $(CFLAGS) $< $(CUDA_OBJS_DEBUG) $(CUDA_C_OBJS_DEBUG) -o $@ $(CLIBS) && echo "CUDA debug build complete! Run the program with: $@"

clean:
	@echo "Cleaning previous build..."
	@$(RM) $(MAIN_BIN)
	@$(RM) $(DEBUG_BIN)
	@$(RM) $(X86_64_V1_BIN)
	@$(RM) $(X86_64_V2_BIN)
	@$(RM) $(X86_64_V3_BIN)
	@$(RM) $(X86_64_V4_BIN)
	@$(RM) -r $(OBJ_DIR)
	@$(RM) $(CUDA_BIN)
	@$(RM) $(CUDA_DEBUG_BIN)
	@$(RM) -r $(CUDA_OBJ_DIR)
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

check-setup: $(RESULTS_DIR) $(SETUP_COMPLETE)

setup:
	@echo "Setting up required tools..."
	@$(RM) $(SETUP_COMPLETE)
	@$(MAKE) check-setup

$(CUDA_COMPLETE): | $(META_DIR)
ifneq ($(IS_WINDOWS),yes)
	@echo "Checking CUDA toolkit..."
	@which nvcc > /dev/null 2>&1 || (echo "Error: CUDA toolkit not found. Please install CUDA toolkit." && exit 1)
	@touch $@
endif

check-cuda: check-setup $(CUDA_COMPLETE)
ifeq ($(IS_WINDOWS),yes)
	@echo "Error: CUDA builds are not supported on Windows for now." && exit 1
endif

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