IS_WINDOWS := $(if $(filter Windows_NT,$(OS)),yes,)
IS_CROSS := $(if $(filter cross,$(MAKECMDGOALS)),1,)
RM := $(if $(IS_WINDOWS),powershell -Command "Remove-Item -Force -ErrorAction SilentlyContinue",rm -f)

CC := $(if $(IS_CROSS),x86_64-w64-mingw32-gcc,gcc)
BIN_EXT := $(if $(or $(IS_CROSS),$(IS_WINDOWS)),.exe,)

MAIN_SRC := src/main.c #$(wildcard src/*.c)
TUNER_SRC := src/tuners/batch.c #$(wildcard src/tuners/*.c)
MAIN_BINS := $(patsubst src/%.c,bin/%$(BIN_EXT),$(MAIN_SRC))
TUNER_BINS := $(patsubst src/tuners/%.c,bin/%$(BIN_EXT),$(TUNER_SRC))

IS_W64DEVKIT := $(if $(IS_WINDOWS),$(if $(findstring w64devkit,$(shell where gcc $(if $(IS_WINDOWS),2>nul,2>/dev/null))),yes,),)

BASE_FLAGS := -march=native -pthread -Iinclude $(if $(IS_CROSS),-DCROSS_COMPILE,)
OPT_FLAGS := -O3 -ffast-math -funroll-loops -fno-strict-aliasing \
             -fprefetch-loop-arrays "-Wl,--gc-sections" -DNDEBUG \
             $(if $(IS_W64DEVKIT),,-flto)
DBG_FLAGS := -g -O0 -Wall -Wextra -Wpedantic -Werror -fstack-protector-strong

CFLAGS := $(BASE_FLAGS) $(if $(filter debug,$(MAKECMDGOALS)),$(DBG_FLAGS),$(OPT_FLAGS))

LIBS := -lpthread $(if $(IS_WINDOWS),-lShlwapi,) $(if $(IS_CROSS),-lshlwapi,)

.PHONY: all debug tune cross dataset clean

all: bin clean-main $(MAIN_BINS)

debug: bin clean-main $(MAIN_BINS)

tune: bin clean-tuner $(TUNER_BINS)

cross: all

dataset: testing/datasets/avpdb.csv
	python3 scripts/create_mega_dataset.py -sc

clean: clean-main clean-tuner

bin:
	$(if $(IS_WINDOWS),powershell -Command "if (-not (Test-Path bin)) { New-Item -ItemType Directory -Path bin | Out-Null }",mkdir -p bin)

bin/%$(BIN_EXT): src/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

bin/%$(BIN_EXT): src/tuners/%.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean-main:
	$(if $(IS_WINDOWS), $(foreach bin,$(MAIN_BINS),$(RM) $(bin);) exit 0;, $(RM) $(MAIN_BINS))
	$(if $(IS_WINDOWS),, $(RM) $(patsubst %,%.exe,$(MAIN_BINS)))

clean-tuner:
	$(if $(IS_WINDOWS), $(foreach bin,$(TUNER_BINS),$(RM) $(bin);) exit 0;, $(RM) $(TUNER_BINS))