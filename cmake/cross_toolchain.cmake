set(PKG_CONFIG_EXECUTABLE "x86_64-w64-mingw32-pkg-config" CACHE FILEPATH "pkg-config for cross-compilation")

execute_process(COMMAND ${PKG_CONFIG_EXECUTABLE} --variable=prefix hdf5 OUTPUT_VARIABLE MINGW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT MINGW_PREFIX)
	message(FATAL_ERROR "Could not determine MinGW prefix via pkg-config")
endif()

set(MINGW_PREFIX "${MINGW_PREFIX}" CACHE PATH "MinGW prefix discovered via pkg-config")

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_FIND_ROOT_PATH "${MINGW_PREFIX}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(IS_CROSS ON CACHE BOOL "Cross-compiling for Windows")
