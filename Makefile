MAKEFLAGS += --no-print-directory -j$(shell nproc)
.PHONY: native cuda help clean update setup debug cuda-debug all check-setup check-libraries check-cuda _x86-64-v1 _x86-64-v2 _x86-64-v3 _x86-64-v4

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
	@echo "  $(MAKE_CMD) help         - Show this help message"
	@echo "  $(MAKE_CMD) native       - Build the program for your specific PC"
ifneq ($(IS_WINDOWS),yes)
	@echo "  $(MAKE_CMD) cuda         - Build with CUDA support and for your specific PC"
endif
	@echo "  $(MAKE_CMD) cross        - Build the program for Windows from another OS"
	@echo "  $(MAKE_CMD) clean        - Remove built files"
	@echo ""
	@echo "After building, run the program with: $(MAIN_BIN)"

# TODO: Check if running MSYS2 from windows terminal and replace RM and MKDIR with:
#RM := $(if $(IS_WINDOWS),powershell -Command "Remove-Item -Force -ErrorAction SilentlyContinue",rm -f)
#$(if $(IS_WINDOWS),powershell -Command "if (-not (Test-Path $(FILE))) { New-Item -ItemType Directory -Path $(FILE) | Out-Null }",mkdir -p $(FILE))
# or move it to a simple .sh script
RM := rm -f
MKDIR := mkdir -p
check_dir_exists = $(if $(wildcard $1),,$(error Error: $2 directory ($1) not found! Please don't move the Makefile without the required directories))

ifneq ($(filter cross cross-debug,$(MAKECMDGOALS)),)
IS_CROSS_COMPILE := yes
endif

BIN_DIR := bin
OBJ_DIR := $(BIN_DIR)/obj$(if $(IS_CROSS_COMPILE),-cross,)

SRC_DIR := code/src
$(call check_dir_exists,$(SRC_DIR),Source)
INCLUDE_DIR := code/include
$(call check_dir_exists,$(INCLUDE_DIR),Include)
SCRIPTS_DIR := scripts
$(call check_dir_exists,$(SCRIPTS_DIR),Scripts)

MAIN_SRC := $(SRC_DIR)/main.c
HEADERS := $(wildcard $(INCLUDE_DIR)/*.h)
rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))
SRCS := $(filter-out $(MAIN_SRC),$(call rwildcard,$(SRC_DIR)/,*.c))
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
OBJS_DEBUG := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-debug.o,$(SRCS))
OBJS_X86_64_V1 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-x86-64-v1.o,$(SRCS))
OBJS_X86_64_V2 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-x86-64-v2.o,$(SRCS))
OBJS_X86_64_V3 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-x86-64-v3.o,$(SRCS))
OBJS_X86_64_V4 := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%-x86-64-v4.o,$(SRCS))


CC := gcc
ifeq ($(IS_WINDOWS),yes)
CC := mingw32-gcc
else ifeq ($(IS_CROSS_COMPILE),yes)
CC := x86_64-w64-mingw32-gcc
MINGW_PKG_CONFIG := x86_64-w64-mingw32-pkg-config
endif
BIN_EXT := $(if $(IS_WINDOWS),.exe,$(if $(IS_CROSS_COMPILE),.exe,))
MAIN_BIN := $(BIN_DIR)/seqalign$(BIN_EXT)
DEBUG_BIN := $(BIN_DIR)/seqalign-debug$(BIN_EXT)
X86_64_V1_BIN := $(BIN_DIR)/seqalign-x86-64-v1$(BIN_EXT)
X86_64_V2_BIN := $(BIN_DIR)/seqalign-x86-64-v2$(BIN_EXT)
X86_64_V3_BIN := $(BIN_DIR)/seqalign-x86-64-v3$(BIN_EXT)
X86_64_V4_BIN := $(BIN_DIR)/seqalign-x86-64-v4$(BIN_EXT)

HDF5_CFLAGS := $(if $(IS_WINDOWS),-I/ucrt64/include,$(if $(IS_CROSS_COMPILE),$(shell $(MINGW_PKG_CONFIG) --cflags hdf5),$(shell bash $(SCRIPTS_DIR)/check_hdf5.sh --parseable | head -n 1)))
BASE_CFLAGS := -I$(INCLUDE_DIR) $(HDF5_CFLAGS) -std=c2x $(if $(filter yes,$(IS_WINDOWS) $(IS_CROSS_COMPILE)),-DWIN32_LEAN_AND_MEAN,-D_GNU_SOURCE -pthread)
RELEASE_CFLAGS := $(BASE_CFLAGS) -O3 \
                  -funroll-loops -fprefetch-loop-arrays \
                  -fdata-sections -ffunction-sections \
                  -fPIC -fno-plt -DNDEBUG
RELEASE_LDFLAGS := -flto "-Wl,--gc-sections" "-Wl,-O3" "-Wl,--as-needed"
DEBUG_CFLAGS := $(BASE_CFLAGS) -g -O0 -Wall -Wextra -Werror \
                -Wshadow -Wconversion -Wmissing-declarations \
                -Wundef -Wfloat-equal -Wcast-align \
                -Wstrict-prototypes -Wswitch-enum -pedantic \
                -fstack-protector-strong -fanalyzer $(if $(filter yes,$(IS_WINDOWS) $(IS_CROSS_COMPILE)),,-fsanitize=address)


HDF5_CLIBS := $(if $(IS_WINDOWS),-L/ucrt64/lib -lhdf5 -lz,$(if $(IS_CROSS_COMPILE),$(shell $(MINGW_PKG_CONFIG) --libs hdf5) -lz,$(shell bash $(SCRIPTS_DIR)/check_hdf5.sh --parseable | tail -n 1)))
CLIBS := $(HDF5_CLIBS) $(if $(IS_WINDOWS),-lShlwapi,)


CUDA_OBJ_DIR := $(BIN_DIR)/cuda_obj
ifneq ($(filter cuda cuda-debug,$(MAKECMDGOALS)),)
CUDA_SRC_DIR := code/cuda/src
$(call check_dir_exists,$(CUDA_SRC_DIR),CUDA source)
CUDA_INCLUDE_DIR := code/cuda/include
$(call check_dir_exists,$(CUDA_INCLUDE_DIR),CUDA include)
CUDA_C_BINDINGS_DIR := code/cuda/c_bindings

CUDA_SRCS := $(wildcard $(CUDA_SRC_DIR)/*.cu)
CUDA_HEADERS := $(wildcard $(CUDA_INCLUDE_DIR)/*.{h,hpp,cuh}) $(wildcard $(CUDA_C_BINDINGS_DIR)/*.{h,hpp,cuh})
HEADERS += $(wildcard $(CUDA_C_BINDINGS_DIR)/*.{h,hpp,cuh})
CUDA_OBJS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_SRCS))
CUDA_C_OBJS := $(patsubst $(SRC_DIR)/%.c,$(CUDA_OBJ_DIR)/%.o,$(SRCS))
CUDA_OBJS_DEBUG := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%-debug.o,$(CUDA_SRCS))
CUDA_C_OBJS_DEBUG := $(patsubst $(SRC_DIR)/%.c,$(CUDA_OBJ_DIR)/%-debug.o,$(SRCS))

NVCC := nvcc
CUDA_RELEASE_FLAGS := -std=c++20 \
    --extra-device-vectorization \
    -Xcompiler="-O3 -fPIC -fno-plt" \
    -I$(CUDA_INCLUDE_DIR) -I$(CUDA_C_BINDINGS_DIR)
CUDA_DEBUG_FLAGS := -g -G -O0 -Xcompiler "-Wall -Wextra" -I$(CUDA_INCLUDE_DIR) -I$(CUDA_C_BINDINGS_DIR) -Wno-deprecated-gpu-targets

CUDA_LIBS := $(shell pkg-config --libs cuda) -lcudart
CUDA_CFLAGS := -DUSE_CUDA -I$(CUDA_C_BINDINGS_DIR)
endif
CUDA_BIN := $(BIN_DIR)/seqalign-cuda$(BIN_EXT)
CUDA_DEBUG_BIN := $(BIN_DIR)/seqalign-cuda-debug$(BIN_EXT)

native: CFLAGS := -march=native $(RELEASE_CFLAGS)
native: LDFLAGS := $(RELEASE_LDFLAGS)
native: $(MAIN_BIN)
cross: native
all: native

debug: CFLAGS := -march=native $(DEBUG_CFLAGS)
debug: LDFLAGS :=
debug: $(DEBUG_BIN)
cross-debug: debug

# See: https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels for more details

# baseline for all x86-64 CPUs, Intel Prescott (2004+) and AMD K8 (2003+)
_x86-64-v1: CFLAGS := -march=x86-64 $(RELEASE_CFLAGS)
_x86-64-v1: LDFLAGS := $(RELEASE_LDFLAGS)
_x86-64-v1: $(X86_64_V1_BIN)

# Intel Nehalem (2008+) and (Atom) Silvermont (SoC) (2013+), AMD Bulldozer (2011+) and Jaguar (2013+)
_x86-64-v2: CFLAGS := -march=x86-64-v2 $(RELEASE_CFLAGS)
_x86-64-v2: LDFLAGS := $(RELEASE_LDFLAGS)
_x86-64-v2: $(X86_64_V2_BIN)

# Intel Haswell (2013+) with AVX2 enabled models and (Atom) Gracemont (SoC) (2021+), AMD Excavator (2015+) and Zen 1+ (2017+)
_x86-64-v3: CFLAGS := -march=x86-64-v3 $(RELEASE_CFLAGS)
_x86-64-v3: LDFLAGS := $(RELEASE_LDFLAGS)
_x86-64-v3: $(X86_64_V3_BIN)

# Intel Skylake (2015+) with AVX512 enabled models, AMD Zen 4+ (2022+)
_x86-64-v4: CFLAGS := -march=x86-64-v4 $(RELEASE_CFLAGS)
_x86-64-v4: LDFLAGS := $(RELEASE_LDFLAGS)
_x86-64-v4: $(X86_64_V4_BIN)

cuda: CFLAGS := -march=native $(RELEASE_CFLAGS) $(CUDA_CFLAGS)
cuda: LDFLAGS := $(RELEASE_LDFLAGS)
cuda: CLIBS += $(CUDA_LIBS) -lstdc++
cuda: $(CUDA_BIN)

cuda-debug: CFLAGS := -march=native $(DEBUG_CFLAGS) $(CUDA_CFLAGS) -fno-sanitize=address
cuda-debug: LDFLAGS :=
cuda-debug: CLIBS += $(CUDA_LIBS) -lstdc++
cuda-debug: $(CUDA_DEBUG_BIN)

$(BIN_DIR) $(OBJ_DIR) $(CUDA_OBJ_DIR):
	@$(MKDIR) $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "$<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-debug.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "$<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-x86-64-v1.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "$<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-x86-64-v2.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "$<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-x86-64-v3.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "$<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%-x86-64-v4.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "$<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(MAIN_BIN): $(MAIN_SRC) $(OBJS) $(HEADERS) | $(BIN_DIR) check-setup
	@if [ -L "$(MAIN_BIN)" ]; then \
        echo "Removing existing symlink at $(MAIN_BIN)"; \
        $(RM) $(MAIN_BIN); \
    fi
	@echo "$<"
	@$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(CLIBS) $(LDFLAGS) && echo "Build complete! Run the program with: $@"

define BINARY_RULE
$(1): $(MAIN_SRC) $(2) $(HEADERS) | $(BIN_DIR) check-setup
	@echo "$$<"
	@$$(CC) $$(CFLAGS) $$< $(2) -o $$@ $$(CLIBS) $$(LDFLAGS) && echo "$(3)"
endef

$(eval $(call BINARY_RULE,$(DEBUG_BIN),$(OBJS_DEBUG),Debug build complete! Run the program with: $$@))
$(eval $(call BINARY_RULE,$(X86_64_V1_BIN),$(OBJS_X86_64_V1),x86-64-v1 build complete!))
$(eval $(call BINARY_RULE,$(X86_64_V2_BIN),$(OBJS_X86_64_V2),x86-64-v2 build complete!))
$(eval $(call BINARY_RULE,$(X86_64_V3_BIN),$(OBJS_X86_64_V3),x86-64-v3 build complete!))
$(eval $(call BINARY_RULE,$(X86_64_V4_BIN),$(OBJS_X86_64_V4),x86-64-v4 build complete!))

$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu $(CUDA_HEADERS) | $(CUDA_OBJ_DIR)
	@echo "$<"
	@$(NVCC) $(CUDA_RELEASE_FLAGS) -c $< -o $@

$(CUDA_OBJ_DIR)/%-debug.o: $(CUDA_SRC_DIR)/%.cu $(CUDA_HEADERS) | $(CUDA_OBJ_DIR)
	@echo "$<"
	@$(NVCC) $(CUDA_DEBUG_FLAGS) -c $< -o $@

$(CUDA_OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(CUDA_OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "$<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(CUDA_OBJ_DIR)/%-debug.o: $(SRC_DIR)/%.c $(HEADERS) | $(CUDA_OBJ_DIR)
	@$(MKDIR) $(dir $@)
	@echo "$<"
	@$(CC) $(CFLAGS) -c $< -o $@

$(CUDA_BIN): $(MAIN_SRC) $(HEADERS) $(CUDA_OBJS) $(CUDA_C_OBJS) $(CUDA_HEADERS) | $(BIN_DIR) check-cuda
	@echo "$<"
	@$(CC) $(CFLAGS) $< $(CUDA_OBJS) $(CUDA_C_OBJS) -o $@ $(CLIBS) $(LDFLAGS) && echo "CUDA build complete! Run the program with: $@"
	@if [ ! -f "$(MAIN_BIN)" ] && [ -f "$(CUDA_BIN)" ]; then \
        echo "Creating symlink from $(MAIN_BIN) to $(CUDA_BIN)"; \
        ln -sf $(notdir $(CUDA_BIN)) $(MAIN_BIN); \
    fi

$(CUDA_DEBUG_BIN): $(MAIN_SRC) $(HEADERS) $(CUDA_OBJS_DEBUG) $(CUDA_C_OBJS_DEBUG) $(CUDA_HEADERS) | $(BIN_DIR) check-cuda
	@echo "$<"
	@$(CC) $(CFLAGS) $< $(CUDA_OBJS_DEBUG) $(CUDA_C_OBJS_DEBUG) -o $@ $(CLIBS) $(LDFLAGS) && echo "CUDA debug build complete! Run the program with: $@"

clean:
	@echo "Cleaning build files..."
	@$(RM) -rf $(BIN_DIR)

check-setup:
ifeq ($(IS_WINDOWS),yes)
	@bash $(SCRIPTS_DIR)/check_msys2.sh
else ifeq ($(IS_CROSS_COMPILE),yes)
	@bash $(SCRIPTS_DIR)/check_cross.sh
else
	@which $(CC) >/dev/null 2>&1 || (echo "Error: Compiler not found. Please install GCC." && exit 1)
	@which pkg-config >/dev/null 2>&1 || (echo "Error: pkg-config not found. Please install pkg-config." && exit 1)
	@bash $(SCRIPTS_DIR)/check_hdf5.sh --quiet || echo "Warning: Required packages not found. Build might fail."
endif

check-cuda: check-setup
ifeq ($(IS_WINDOWS),yes)
	@echo "Error: Use MSVC NMakefile to compile with CUDA support." && exit 1
else
	@which nvcc > /dev/null 2>&1 || (echo "Error: CUDA toolkit not found. Please install CUDA toolkit." && exit 1)
endif
