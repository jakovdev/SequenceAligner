cmake_minimum_required(VERSION 3.30)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
project(SequenceAligner C)

set(ARCH_LEVEL "native" CACHE STRING "Architecture level")

set(IS_MINGW FALSE)
set(IS_MSYS2 FALSE)
set(IS_MSVC FALSE)
set(IS_GNU FALSE)
set(IS_GWIN FALSE)

if(MINGW_PREFIX)
	set(IS_MINGW TRUE)
	set(IS_GNU TRUE)
	set(IS_GWIN TRUE)
	set(DLL_PREFIX "${MINGW_PREFIX}")
	set(DLL_LIST libhdf5.dll libssp-0.dll libwinpthread-1.dll libsz.dll zlib1.dll libgomp-1.dll libgcc_s_seh-1.dll)
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
		printf("AVX512");
	else if (avx2 && avx && has_os_avx_state())
		printf("AVX2");
	else if (avx && has_os_avx_state())
		printf("AVX");
	else if (sse42)
		printf("SSE4.2");
	else
		printf("SSE2");
	return 0;
}
		]=])
		try_run(ARCH_DETECT_RUN_RES ARCH_DETECT_COMPILE_RES
		SOURCE_FROM_CONTENT arch_probe.c "${_arch_probe_src}"
		RUN_OUTPUT_VARIABLE ARCH_LEVEL
		)
		if(NOT ARCH_DETECT_COMPILE_RES OR NOT ARCH_DETECT_RUN_RES EQUAL 0 OR ARCH_LEVEL STREQUAL "")
			message(WARNING "Native arch detection failed, falling back to SSE2")
			set(ARCH_LEVEL "SSE2")
		else()
			message(STATUS "Detected native arch: ${ARCH_LEVEL}")
		endif()
	endif()
elseif(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "GNU")
	set(IS_MSYS2 TRUE)
	set(IS_GNU TRUE)
	set(IS_GWIN TRUE)
	set(DLL_PREFIX "$ENV{MSYSTEM_PREFIX}")
	file(GLOB HDF5_DLL "${DLL_PREFIX}/bin/libhdf5-[0-9]*.dll")
	string(REGEX REPLACE ".+/(libhdf5-[0-9]+\\.dll)" "\\1" HDF5_DLL "${HDF5_DLL}")
	set(DLL_LIST ${HDF5_DLL}
	libcrypto-3-x64.dll libwinpthread-1.dll libcurl-4.dll libsz-2.dll zlib1.dll libgomp-1.dll
	libbrotlidec.dll libidn2-0.dll libnghttp2-14.dll libnghttp3-9.dll libpsl-5.dll libssh2-1.dll
	libbrotlicommon.dll libzstd.dll libssl-3-x64.dll libintl-8.dll libiconv-2.dll libunistring-5.dll
	libgcc_s_seh-1.dll libngtcp2-16.dll libngtcp2_crypto_ossl-0.dll libaws-c-io.dll libaws-c-s3.dll
	libaws-c-auth.dll libaws-c-http.dll libaws-c-common.dll libaws-c-cal.dll libaws-c-sdkutils.dll
	libaws-c-compression.dll libaws-checksums.dll
	)
elseif(UNIX AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
	set(IS_GNU TRUE)
else()
	message(WARNING "Unknown platform/compiler combination. Defaulting to GNU settings.")
	set(IS_GNU TRUE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(USE_CUDA FALSE)
if(NOT IS_GWIN)
	find_package(CUDAToolkit)
	if(CUDAToolkit_FOUND)
		enable_language(CUDA)
		set(CMAKE_CUDA_STANDARD 20)
		set(CMAKE_CUDA_STANDARD_REQUIRED ON)
		set(USE_CUDA TRUE)
	endif()
endif()

if(IS_GWIN)
	find_package(PkgConfig REQUIRED)
	pkg_check_modules(HDF5 REQUIRED hdf5)
else()
	find_package(HDF5 REQUIRED COMPONENTS C)
endif()

find_package(OpenMP REQUIRED COMPONENTS C)

file(GLOB_RECURSE C_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.c")
add_executable(seqalign ${C_SOURCES})

if(USE_CUDA)
	file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cu")
	target_sources(seqalign PRIVATE ${CUDA_SOURCES})
endif()

target_include_directories(seqalign PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party/clix)

if(IS_GWIN)
	set(TINYCTHREAD_DIR "${CMAKE_SOURCE_DIR}/third_party/tinycthread")
	target_include_directories(seqalign PRIVATE ${TINYCTHREAD_DIR})
	target_sources(seqalign PRIVATE ${TINYCTHREAD_DIR}/sources/tinycthread.c)
endif()

target_compile_definitions(seqalign PRIVATE
$<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:${USE_CUDA}>>:USE_CUDA>
$<$<BOOL:${IS_MSVC}>:_CRT_SECURE_NO_WARNINGS>
$<$<BOOL:${IS_GNU}>:_GNU_SOURCE>
)
target_compile_options(seqalign PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:-cudart=static>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
$<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:${IS_MSVC}>>:/W4;/utf-8;/arch:${ARCH_LEVEL};/experimental:c11atomics;/GF;/Gy;/GA;/Gw;/jumptablerdata>
$<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:${IS_MSVC}>,$<CONFIG:Release>>:/Ox;/GL>
$<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:${IS_GNU}>>:-Wall;-Wextra;-march=${ARCH_LEVEL};-funroll-loops;-fwrapv;-fno-plt;-fvisibility=hidden;-malign-data=cacheline;-flto>
$<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:${IS_GNU}>,$<CONFIG:Debug>>:-Og>
)
target_link_options(seqalign PRIVATE
$<$<BOOL:${IS_MSVC}>:/OPT:REF>
$<$<AND:$<BOOL:${IS_MSVC}>,$<CONFIG:Release>>:/LTCG;/OPT:ICF>
$<$<BOOL:${IS_GNU}>:-flto>
)

if(IS_GWIN)
	target_include_directories(seqalign PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(seqalign PRIVATE ${HDF5_LIBRARIES})
	target_link_directories(seqalign PRIVATE ${HDF5_LIBRARY_DIRS})
else()
	target_link_libraries(seqalign PRIVATE hdf5::hdf5)
endif()

if(IS_MSVC)
	target_link_libraries(seqalign PRIVATE Shlwapi)
elseif(IS_GWIN)
	target_link_libraries(seqalign PRIVATE shlwapi)
endif()

if(USE_CUDA)
	target_link_libraries(seqalign PRIVATE CUDA::cudart_static)
endif()

target_link_libraries(seqalign PRIVATE OpenMP::OpenMP_C)

if(IS_MSVC)
	install(TARGETS seqalign 
	RUNTIME_DEPENDENCIES
	PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "vcruntime.*\\.dll"
	POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
	RUNTIME DESTINATION .
	)
elseif(IS_GWIN)
	foreach(dll ${DLL_LIST})
		if(EXISTS "${DLL_PREFIX}/bin/${dll}")
			install(FILES "${DLL_PREFIX}/bin/${dll}" DESTINATION .)
		elseif(EXISTS "${DLL_PREFIX}/lib/${dll}")
			install(FILES "${DLL_PREFIX}/lib/${dll}" DESTINATION .)
		else()
			message(WARNING "Could not find runtime DLL ${dll} in expected locations under prefix ${DLL_PREFIX}")
		endif()
	endforeach()
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
	if(NOT COMPILE_COMMANDS_LINK_RESULT STREQUAL "0" AND NOT EXISTS "${COMPILE_COMMANDS_ROOT}")
		message(WARNING "Failed to link compile_commands.json to project root: ${COMPILE_COMMANDS_LINK_RESULT}")
	endif()
endif()

set(CPACK_PACKAGE_NAME "SequenceAligner")
set(CPACK_PACKAGE_VENDOR "Jakov Dragičević")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance all-vs-all pairwise sequence alignment tool")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/release")

if(WIN32)
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-windows-${ARCH_LEVEL}")
	set(CPACK_GENERATOR "ZIP")
else()
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-linux-${ARCH_LEVEL}")
	set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)
