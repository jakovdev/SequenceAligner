cmake_minimum_required(VERSION 3.30)

set(ARCH_LEVEL "native" CACHE STRING "Architecture level")
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_ASAN "Enable AddressSanitizer (ASAN) support (GCC|Clang only, no CUDA)" OFF)
option(USE_BIN "Copy binaries to bin directory for easier execution during development" OFF)
option(USE_MINGW "Enable Linux to Windows MinGW build" OFF)

if(USE_MINGW)
	if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
		message(FATAL_ERROR "USE_MINGW=ON is only supported on Linux host")
	endif()
	if(DEFINED ENV{MSYSTEM})
		message(FATAL_ERROR "USE_MINGW=ON should not be used from MSYS2, configure normally with native tools")
	endif()
	find_program(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc REQUIRED)
	find_program(PKG_CONFIG_EXECUTABLE x86_64-w64-mingw32-pkg-config REQUIRED)
	execute_process(COMMAND "${PKG_CONFIG_EXECUTABLE}" --variable=prefix hdf5 OUTPUT_VARIABLE MINGW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
	if(NOT MINGW_PREFIX)
		message(FATAL_ERROR "Could not determine MinGW prefix via pkg-config")
	endif()
	set(MINGW_PREFIX "${MINGW_PREFIX}" CACHE PATH "MinGW prefix discovered via pkg-config")
	set(CMAKE_SYSTEM_NAME Windows)
	set(CMAKE_FIND_ROOT_PATH "${MINGW_PREFIX}")
	set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
	set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
	set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
endif()

if(USE_CUDA)
	set(SEQALIGN_LANGUAGES "C;CUDA")
else()
	set(SEQALIGN_LANGUAGES "C")
endif()

project(
SequenceAligner
VERSION 1.0
DESCRIPTION "High performance all-vs-all pairwise sequence alignment tool"
HOMEPAGE_URL "https://www.github.com/jakovdev/SequenceAligner"
LANGUAGES ${SEQALIGN_LANGUAGES}
)

set(IS_MINGW FALSE)
set(IS_MSVC FALSE)
set(IS_MSYS2 FALSE)
set(IS_GNU FALSE)
set(IS_GCC7 FALSE)
set(IS_GCC10 FALSE)

if(USE_MINGW)
	set(IS_MINGW TRUE)
elseif(MSVC)
	set(IS_MSVC TRUE)
	if(MSVC_VERSION LESS 1939)
		message(FATAL_ERROR "MSVC version 1939 or greater is required (Visual Studio 2022 17.9+). Current version: ${MSVC_VERSION}")
	endif()
	if(ARCH_LEVEL STREQUAL "native")
		set(_arch_probe_src [=[
#include <stdio.h>
#include <intrin.h>
static int has_os_avx_state(void) {
	int cpuInfo[4] = {0};
	__cpuid(cpuInfo, 1);
	const int osxsave = (cpuInfo[2] & (1 << 27)) != 0;
	if (!osxsave) return 0;
	unsigned long long xcr0 = _xgetbv(0);
	return (xcr0 & 0x6) == 0x6; /* XMM+YMM */
}
static int has_avx512_state(void) {
	unsigned long long xcr0 = _xgetbv(0);
	return (xcr0 & 0xE6) == 0xE6; /* XMM+YMM+ZMM state enabled */
}
int main(void) {
	int c[4] = {0};
	__cpuid(c, 1);
	const int sse42 = (c[2] & (1 << 20)) != 0;
	const int avx   = (c[2] & (1 << 28)) != 0;
	__cpuidex(c, 7, 0);
	const int avx2   = (c[1] & (1 << 5))  != 0;
	const int avx512 = (c[1] & (1 << 16)) != 0; /* AVX-512F */
	if (avx512 && has_os_avx_state() && has_avx512_state())
		puts("AVX512");
	else if (avx2 && avx && has_os_avx_state())
		puts("AVX2");
	else if (avx && has_os_avx_state())
		puts("AVX");
	else if (sse42)
		puts("SSE4.2");
	else
		puts("SSE2");
	return 0;
}
		]=])
		try_run(ARCH_DETECT_RUN_RES ARCH_DETECT_COMPILE_RES
		SOURCE_FROM_CONTENT arch_probe.c "${_arch_probe_src}"
		RUN_OUTPUT_VARIABLE ARCH_LEVEL
		)
		string(STRIP "${ARCH_LEVEL}" ARCH_LEVEL)
		if(NOT ARCH_DETECT_COMPILE_RES OR NOT ARCH_DETECT_RUN_RES EQUAL 0 OR ARCH_LEVEL STREQUAL "")
			message(WARNING "Native arch detection failed, falling back to SSE2")
			set(ARCH_LEVEL "SSE2")
		else()
			message(STATUS "Detected native arch: ${ARCH_LEVEL}")
		endif()
	endif()
elseif(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "GNU")
	set(IS_MSYS2 TRUE)
elseif(UNIX AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
	set(IS_GNU TRUE)
else()
	message(WARNING "Unknown platform/compiler combination. Defaulting to GNU settings.")
	set(IS_GNU TRUE)
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
	if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 7)
		set(IS_GCC7 TRUE)
	endif()
	if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
		set(IS_GCC10 TRUE)
	endif()
endif()

if(USE_CUDA)
	if(USE_ASAN)
		message(FATAL_ERROR "USE_CUDA=ON is not supported with USE_ASAN=ON")
	endif()
	if(USE_MINGW)
		message(FATAL_ERROR "USE_CUDA=ON is not supported with USE_MINGW=ON")
	endif()
	if(IS_MSYS2)
		message(FATAL_ERROR "USE_CUDA=ON is not supported on MSYS2")
	endif()
endif()

if(USE_ASAN AND NOT IS_GNU)
	message(FATAL_ERROR "AddressSanitizer (ASAN) is only supported on Linux with GCC/Clang")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(USE_CUDA)
	set(CMAKE_CUDA_STANDARD 20)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(IS_MINGW OR IS_MSYS2)
	find_package(Git REQUIRED)
	include(FetchContent)
	cmake_policy(SET CMP0169 OLD) # Needed for automatic patching
	FetchContent_Declare(
		tinycthread
		GIT_REPOSITORY https://github.com/tinycthread/tinycthread.git
		GIT_TAG 6957fc8383d6c7db25b60b8c849b29caab1caaee
		GIT_SHALLOW TRUE
		PATCH_COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> apply "${CMAKE_SOURCE_DIR}/script/tinycthread-mingw.patch"
	)
	FetchContent_GetProperties(tinycthread)
	if(NOT tinycthread_POPULATED)
		FetchContent_Populate(tinycthread)
	endif()
	set(TINYCTHREAD_SOURCE_DIR "${tinycthread_SOURCE_DIR}/source")
	set(TINYCTHREAD_SOURCES "${TINYCTHREAD_SOURCE_DIR}/tinycthread.c")
	set(USING_TINYCTHREAD TRUE)
endif()

file(GLOB_RECURSE C_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.c")
if(USING_TINYCTHREAD)
	list(APPEND C_SOURCES ${TINYCTHREAD_SOURCES})
endif()
if(USE_CUDA)
	file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cu")
endif()

if(IS_MINGW OR IS_MSYS2)
	find_package(PkgConfig REQUIRED)
	pkg_check_modules(HDF5 REQUIRED hdf5)
else()
	set(LIB_TYPE SHARED)
	string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)
	find_package(HDF5 NAMES hdf5 CONFIG QUIET COMPONENTS C ${SEARCH_TYPE})
	if(NOT HDF5_C_${LIB_TYPE}_LIBRARY)
		find_package(HDF5 MODULE REQUIRED COMPONENTS C)
		set(LEGACY_HDF5 TRUE)
	else()
		set(LINK_LIBS ${LINK_LIBS} ${HDF5_C_${LIB_TYPE}_LIBRARY})
		set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}")
	endif()
endif()

if(USE_CUDA)
	find_package(CUDAToolkit REQUIRED)
endif()

find_package(OpenMP REQUIRED COMPONENTS C)

function(set_c_compiler_options target)
if(IS_MSVC)
	string(REPLACE "/Ob2" "/Ob3" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
	string(REPLACE "-Xcompiler=\"-O2 -Ob2\"" "-Xcompiler=/Ob3" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
	target_compile_options(${target} PRIVATE
	$<$<COMPILE_LANGUAGE:C>:/arch:${ARCH_LEVEL};/experimental:c11atomics;/Zc:preprocessor;/utf-8;/TC;/WX;/Wall;/analyze;/wd4710;/wd4711;/wd4820;/wd5045;/wd6993;/wd28301>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Release>>:/Ox;/GL;/GA;/Gy;/Gw;/jumptablerdata>
	)
	target_link_options(${target} PRIVATE /WX $<$<CONFIG:Release>:/LTCG;/OPT:REF;/OPT:ICF>)
	target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(IS_GNU OR IS_MINGW OR IS_MSYS2)
	target_compile_options(${target} PRIVATE
	$<$<COMPILE_LANGUAGE:C>:-march=${ARCH_LEVEL};-fno-strict-overflow;-fwrapv;-pedantic-errors;-Wall;-Wcast-qual;-Wconversion;-Wdisabled-optimization;-Wdouble-promotion;-Werror;-Wextra;-Wfloat-equal;-Winit-self;-Wmissing-declarations;-Wmissing-include-dirs;-Wshadow;-Wstrict-overflow=5;-Wstrict-prototypes;-Wswitch-default;-Wswitch-enum;-Wundef;-Wuninitialized;-Wwrite-strings>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:${IS_GCC7}>>:-Warith-conversion;-Wduplicated-branches;-Wduplicated-cond;-Wlogical-op>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Release>>:-fdata-sections;-ffunction-sections;-fno-pie;-fno-plt;-funroll-loops>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-Og;-g>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>,$<BOOL:${IS_GCC10}>>:-fanalyzer>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>,$<BOOL:${USE_ASAN}>>:-fsanitize=address,undefined,leak,float-divide-by-zero,pointer-compare,pointer-subtract>
	)
	target_link_options(${target} PRIVATE
	$<$<CONFIG:Release>:-flto;-no-pie;-Wl,--as-needed;-Wl,--gc-sections>
	$<$<AND:$<CONFIG:Debug>,$<BOOL:${USE_ASAN}>>:-fsanitize=address,undefined,leak,float-divide-by-zero,pointer-compare,pointer-subtract>
	)
	target_compile_definitions(${target} PRIVATE _GNU_SOURCE)
endif()
endfunction()

function(set_cuda_compiler_options target)
if(NOT USE_CUDA)
	return()
endif()
target_compile_definitions(${target} PRIVATE USE_CUDA)
target_compile_options(${target} PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G;-g>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-extra-device-vectorization>
)
if(IS_MSVC)
	target_compile_options(${target} PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:${ARCH_LEVEL},/Zc:preprocessor,/utf-8,/WX,/Wall,/analyze,/wd4514,/wd4555,/wd4668,/wd4710,/wd4711,/wd4820,/wd5039,/wd5045>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/Od,/MDd,/RTC1,/Zi>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=/GL,/Ox,/MD,/GA,/Gy,/Gw,/DNDEBUG,/jumptablerdata>
	)
elseif(IS_GNU)
	target_compile_options(${target} PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=${ARCH_LEVEL},-pedantic-errors,-Wno-pedantic,-Wall,-Wcast-align,-Wcast-qual,-Wconversion,-Wdisabled-optimization,-Wdouble-promotion,-Werror,-Wextra,-Wfloat-equal,-Winit-self,-Wmissing-declarations,-Wmissing-include-dirs,-Wshadow,-Wstrict-overflow=5,-Wswitch-default,-Wswitch-enum,-Wundef,-Wuninitialized,-Wwrite-strings>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${IS_GCC7}>>:-Xcompiler=-Warith-conversion,-Wduplicated-branches,-Wduplicated-cond,-Wlogical-op>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=-g>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>,$<BOOL:${IS_GCC10}>>:-Xcompiler=-fanalyzer>	
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=-O3,-DNDEBUG,-fdata-sections,-ffunction-sections,-fno-pie,-fno-plt,-funroll-loops>
	)
endif()
endfunction()

add_executable(seqalign ${C_SOURCES})
if(USE_CUDA)
	target_sources(seqalign PRIVATE ${CUDA_SOURCES})
endif()

target_include_directories(seqalign PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(USING_TINYCTHREAD)
	target_include_directories(seqalign PRIVATE ${TINYCTHREAD_SOURCE_DIR})
endif()

set_c_compiler_options(seqalign)
set_cuda_compiler_options(seqalign)

if(LEGACY_HDF5 OR IS_MINGW OR IS_MSYS2)
	target_include_directories(seqalign PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(seqalign PRIVATE ${HDF5_LIBRARIES})
	target_link_directories(seqalign PRIVATE ${HDF5_LIBRARY_DIRS})
else()
	target_link_libraries(seqalign PRIVATE ${LINK_LIBS})
endif()

if(WIN32)
	if(IS_MSVC)
		target_link_libraries(seqalign PRIVATE Shlwapi)
	else()
		target_link_libraries(seqalign PRIVATE shlwapi)
	endif()
endif()

if(USE_CUDA)
	target_link_libraries(seqalign PRIVATE CUDA::cudart)
endif()

target_link_libraries(seqalign PRIVATE OpenMP::OpenMP_C)

function(resolve_runtime_dll_path out_path dll prefix)
if(EXISTS "${prefix}/bin/${dll}")
	set(${out_path} "${prefix}/bin/${dll}" PARENT_SCOPE)
elseif(EXISTS "${prefix}/lib/${dll}")
	set(${out_path} "${prefix}/lib/${dll}" PARENT_SCOPE)
else()
	message(WARNING "Could not find runtime DLL ${dll} in expected locations under prefix ${prefix}")
	set(${out_path} "" PARENT_SCOPE)
endif()
endfunction()

function(install_runtime_dlls dll_list prefix)
foreach(dll ${dll_list})
	resolve_runtime_dll_path(dll_path "${dll}" "${prefix}")
	if(dll_path)
		install(FILES "${dll_path}" DESTINATION .)
	endif()
endforeach()
endfunction()

function(copy_runtime_dlls_to_bin target dll_list prefix output_dir)
foreach(dll ${dll_list})
	resolve_runtime_dll_path(dll_path "${dll}" "${prefix}")
	if(dll_path)
		add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${dll_path}"
		"${output_dir}/"
		)
	endif()
endforeach()
endfunction()

if(WIN32)
	if(IS_MSVC)
		install(TARGETS seqalign 
		RUNTIME_DEPENDENCIES
		PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "vcruntime.*\\.dll"
		POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
		RUNTIME DESTINATION .
		)
	elseif(IS_MSYS2)
		if(NOT DEFINED ENV{MSYSTEM_PREFIX})
			message(FATAL_ERROR "MSYSTEM_PREFIX not set. Cannot locate MSYS2 bin directory.")
		endif()
		file(GLOB HDF5_DLL "$ENV{MSYSTEM_PREFIX}/bin/libhdf5-[0-9]*.dll")
		string(REGEX REPLACE ".+/(libhdf5-[0-9]+\\.dll)" "\\1" HDF5_DLL "${HDF5_DLL}")
		set(MSYS2_DLLS ${HDF5_DLL}
		libcrypto-3-x64.dll libwinpthread-1.dll libcurl-4.dll libsz-2.dll zlib1.dll libgomp-1.dll
		libbrotlidec.dll libidn2-0.dll libnghttp2-14.dll libnghttp3-9.dll libpsl-5.dll libssh2-1.dll
		libbrotlicommon.dll libzstd.dll libssl-3-x64.dll libintl-8.dll libiconv-2.dll libunistring-5.dll
		libgcc_s_seh-1.dll libngtcp2-16.dll libngtcp2_crypto_ossl-0.dll libaws-c-io.dll libaws-c-s3.dll
		libaws-c-auth.dll libaws-c-http.dll libaws-c-common.dll libaws-c-cal.dll libaws-c-sdkutils.dll
		libaws-c-compression.dll libaws-checksums.dll
		)
		install_runtime_dlls("${MSYS2_DLLS}" "$ENV{MSYSTEM_PREFIX}")
	elseif(IS_MINGW)		
		set(MINGW_DLLS libhdf5.dll libssp-0.dll libwinpthread-1.dll libsz.dll zlib1.dll libgomp-1.dll libgcc_s_seh-1.dll)
		install_runtime_dlls("${MINGW_DLLS}" "${MINGW_PREFIX}")
	endif()
endif()

if(USE_BIN)
	set_target_properties(seqalign PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
	file(GLOB BIN_FILES "${CMAKE_SOURCE_DIR}/bin/*")
	if(BIN_FILES)
		file(REMOVE ${BIN_FILES})
	endif()
	if(IS_MSVC)
		add_custom_command(TARGET seqalign POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:seqalign> "${CMAKE_SOURCE_DIR}/bin/")
		install_runtime_dlls("$<TARGET_FILE_DIR:seqalign>" "${CMAKE_SOURCE_DIR}/bin")
	elseif(IS_MSYS2)
		copy_runtime_dlls_to_bin(seqalign "${MSYS2_DLLS}" "$ENV{MSYSTEM_PREFIX}" "${CMAKE_SOURCE_DIR}/bin")
	elseif(IS_MINGW)
		copy_runtime_dlls_to_bin(seqalign "${MINGW_DLLS}" "${MINGW_PREFIX}" "${CMAKE_SOURCE_DIR}/bin")
	endif()
endif()

install(TARGETS seqalign RUNTIME DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" "${CMAKE_SOURCE_DIR}/README.md" DESTINATION .)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
	set(COMPILE_COMMANDS_SOURCE "${CMAKE_BINARY_DIR}/compile_commands.json")
	set(COMPILE_COMMANDS_ROOT "${CMAKE_SOURCE_DIR}/compile_commands.json")
	if(EXISTS "${COMPILE_COMMANDS_ROOT}" OR IS_SYMLINK "${COMPILE_COMMANDS_ROOT}")
		file(REMOVE "${COMPILE_COMMANDS_ROOT}")
	endif()
	file(CREATE_LINK "${COMPILE_COMMANDS_SOURCE}" "${COMPILE_COMMANDS_ROOT}" SYMBOLIC COPY_ON_ERROR RESULT COMPILE_COMMANDS_LINK_RESULT)
	if(NOT COMPILE_COMMANDS_LINK_RESULT STREQUAL "0")
		message(WARNING "Failed to link compile_commands.json to project root: ${COMPILE_COMMANDS_LINK_RESULT}")
	endif()
endif()

set(CPACK_PACKAGE_NAME "SequenceAligner")
set(CPACK_PACKAGE_VENDOR "jakovdev")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance all-vs-all pairwise sequence alignment tool")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/release")

string(TOLOWER "${ARCH_LEVEL}" arch_suffix)
if(USE_CUDA)
	set(arch_suffix "${arch_suffix}-cuda")
endif()

if(WIN32)
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-windows-${arch_suffix}")
	set(CPACK_GENERATOR "ZIP")
else()
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-linux-${arch_suffix}")
	set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)
