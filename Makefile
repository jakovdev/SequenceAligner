IS_WINDOWS := $(if $(filter Windows_NT,$(OS)),yes,)
IS_CROSS := $(if $(filter cross,$(MAKECMDGOALS)),1,)
RM := $(if $(IS_WINDOWS),powershell -Command "Remove-Item -Force -ErrorAction SilentlyContinue",rm -f)

CC := $(if $(IS_CROSS),x86_64-w64-mingw32-gcc,gcc)
BIN_EXT := $(if $(or $(IS_CROSS),$(IS_WINDOWS)),.exe,)

MAIN_SRC := src/main.c #$(wildcard src/*.c)
MAIN_BINS := $(patsubst src/%.c,bin/%$(BIN_EXT),$(MAIN_SRC))

IS_W64DEVKIT := $(if $(IS_WINDOWS),$(if $(findstring w64devkit,$(shell where gcc $(if $(IS_WINDOWS),2>nul,2>/dev/null))),yes,),)

ifeq ($(filter __github_action_test__,$(MAKECMDGOALS)),__github_action_test__)
  HDF5_CFLAGS := -I/usr/include/hdf5/serial
  HDF5_LIBS := -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
else
  HDF5_CFLAGS := $(if $(shell which pkg-config 2>/dev/null && pkg-config --exists hdf5 && echo yes), \
                $(shell pkg-config --cflags hdf5), \
                -I/usr/include/hdf5/serial)
  HDF5_LIBS := $(if $(shell which pkg-config 2>/dev/null && pkg-config --exists hdf5 && echo yes), \
                $(shell pkg-config --libs hdf5), \
                -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5)
endif

BASE_FLAGS := -march=native -pthread -Iinclude $(HDF5_CFLAGS) $(if $(IS_CROSS),-DCROSS_COMPILE,)
OPT_FLAGS := -O3 -ffast-math -funroll-loops -fno-strict-aliasing \
             -fprefetch-loop-arrays "-Wl,--gc-sections" -DNDEBUG \
             $(if $(IS_W64DEVKIT),,-flto)
DBG_FLAGS := -g -O0 -Wall -Wextra -Wpedantic -Werror -fstack-protector-strong

CFLAGS := $(BASE_FLAGS) $(if $(filter debug,$(MAKECMDGOALS)),$(DBG_FLAGS),$(OPT_FLAGS))

LIBS := -lm $(HDF5_LIBS) $(if $(IS_WINDOWS),-lShlwapi,) $(if $(IS_CROSS),-lshlwapi,)

.PHONY: all debug cross clean __github_action_test__

all: bin results clean $(MAIN_BINS)

debug: bin results clean $(MAIN_BINS)

__github_action_test__: bin results clean $(MAIN_BINS)
	@echo "Building for GitHub Actions with hardcoded HDF5 paths"

cross: all

bin:
	$(if $(IS_WINDOWS),powershell -Command "if (-not (Test-Path bin)) { New-Item -ItemType Directory -Path bin | Out-Null }",mkdir -p bin)

results:
	$(if $(IS_WINDOWS),powershell -Command "if (-not (Test-Path results)) { New-Item -ItemType Directory -Path results | Out-Null }",mkdir -p results)

bin/%$(BIN_EXT): src/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	$(if $(IS_WINDOWS), $(foreach bin,$(MAIN_BINS),$(RM) $(bin);) exit 0;, $(RM) $(MAIN_BINS))
	$(if $(IS_WINDOWS),, $(RM) $(patsubst %,%.exe,$(MAIN_BINS)))