cmake_minimum_required(VERSION 3.25)

set(ARCH_LEVEL "native" CACHE STRING "Architecture level")
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_ASAN "Enable AddressSanitizer (ASAN) support (GCC|Clang only, no CUDA)" OFF)
option(USE_BIN "Copy binaries to bin directory for easier execution during development" OFF)

set(SEQALIGN_LANGUAGES "C")
if(USE_CUDA)
	set(SEQALIGN_LANGUAGES "C;CUDA")
endif()

project(
SequenceAligner
VERSION 1.0
DESCRIPTION "High performance all-vs-all pairwise sequence alignment tool"
HOMEPAGE_URL "https://www.github.com/jakovdev/SequenceAligner"
LANGUAGES ${SEQALIGN_LANGUAGES}
)

set(PROJECT_SPDX_LICENSE "AGPL-3.0-or-later")
set(SequenceAligner_SPDX_LICENSE "AGPL-3.0-or-later")

if(IS_CROSS)
	set(PLATFORM_TYPE "CROSS")
	if(NOT MINGW_PREFIX)
		message(FATAL_ERROR "Pass -DCMAKE_TOOLCHAIN_FILE=cmake/cross_toolchain.cmake to enable cross-compilation")
	endif()
elseif(MSVC)
	set(PLATFORM_TYPE "MSVC")
	if(ARCH_LEVEL STREQUAL "native")
		set(ARCH_LEVEL "AVX2")
	endif()
	string(REPLACE "/Ob2" "/Ob3" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
	string(REPLACE "-Xcompiler=\"-O2 -Ob2\"" "-Xcompiler=/Ob3" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
elseif(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "GNU")
	set(PLATFORM_TYPE "MSYS2")
elseif(UNIX AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
	set(PLATFORM_TYPE "GCC")
else()
	message(WARNING "Unknown platform/compiler combination. Defaulting to GCC-like settings.")
	set(PLATFORM_TYPE "GCC")
endif()

if(PLATFORM_TYPE STREQUAL "GCC"
OR PLATFORM_TYPE STREQUAL "CROSS"
OR PLATFORM_TYPE STREQUAL "MSYS2")
	set(IS_GCC_LIKE TRUE)
	include(CheckCCompilerFlag)
	set(GCC_FLAGS_TO_CHECK
	-fprefetch-loop-arrays
	-fanalyzer
	-Warith-conversion
	-Wduplicated-branches
	-Wduplicated-cond
	-Wlogical-op
	-Wopenmp
	-Wuseless-cast
	)
	set(SUPPORTS_GCC_FLAGS TRUE)
	foreach(_flag IN LISTS GCC_FLAGS_TO_CHECK)
		string(REGEX REPLACE "^-+" "" _name "${_flag}")
		string(REGEX REPLACE "[^A-Za-z0-9]" "_" _name "${_name}")
		set(_var "HAVE_GCC_FLAG_${_name}")
		check_c_compiler_flag("${_flag}" ${_var})
		if(NOT ${_var})
			set(SUPPORTS_GCC_FLAGS FALSE)
			break()
		endif()
	endforeach()
	if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		find_program(GDB_EXECUTABLE gdb)
		string(REGEX REPLACE ".+" "TRUE" HAS_GDB "${GDB_EXECUTABLE}")
	endif()
endif()

if(USE_ASAN AND USE_CUDA)
	message(FATAL_ERROR "AddressSanitizer (ASAN) is not supported with CUDA")
endif()

if(USE_ASAN AND NOT PLATFORM_TYPE STREQUAL "GCC")
	message(FATAL_ERROR "AddressSanitizer (ASAN) is only supported on Linux with GCC/Clang")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(USE_CUDA)
	set(CMAKE_CUDA_STANDARD 20)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

file(GLOB_RECURSE C_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/code/src/*.c")
if(USE_CUDA)
	file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/code/cuda/src/*.cu")
endif()

if(IS_CROSS OR PLATFORM_TYPE STREQUAL "MSYS2")
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
if(PLATFORM_TYPE STREQUAL "MSVC")
	target_compile_options(${target} PRIVATE
	$<$<COMPILE_LANGUAGE:C>:/arch:${ARCH_LEVEL}>
	$<$<COMPILE_LANGUAGE:C>:/experimental:c11atomics>
	$<$<COMPILE_LANGUAGE:C>:/Zc:preprocessor>
	$<$<COMPILE_LANGUAGE:C>:/utf-8>
	$<$<COMPILE_LANGUAGE:C>:/TC>
	$<$<COMPILE_LANGUAGE:C>:/WX>
	$<$<COMPILE_LANGUAGE:C>:/Wall>
	$<$<COMPILE_LANGUAGE:C>:/analyze>
	$<$<COMPILE_LANGUAGE:C>:/wd4710> # not inlined info
	$<$<COMPILE_LANGUAGE:C>:/wd4711> # automatic inline info
	$<$<COMPILE_LANGUAGE:C>:/wd4820> # struct padding
	$<$<COMPILE_LANGUAGE:C>:/wd5045> # spectre mitigation
	$<$<COMPILE_LANGUAGE:C>:/wd6993> # analyzer ignores openmp
	$<$<COMPILE_LANGUAGE:C>:/wd28301> # no annotations for first declaration (in Windows 10 SDK)
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Release>>:/Ox>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Release>>:/GL>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/GA>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/Gy>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/Gw>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/jumptablerdata>
	)
	target_link_options(${target} PRIVATE
	/WX
	$<$<CONFIG:Release>:/LTCG>
	$<$<CONFIG:Release>:/OPT:REF>
	$<$<CONFIG:Release>:/OPT:ICF>
	)
	target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN)
	target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(IS_GCC_LIKE)
	target_compile_options(${target} PRIVATE
	$<$<COMPILE_LANGUAGE:C>:-march=${ARCH_LEVEL}>
	$<$<COMPILE_LANGUAGE:C>:-pedantic-errors>
	$<$<COMPILE_LANGUAGE:C>:-Wall>
	$<$<COMPILE_LANGUAGE:C>:-Wcast-qual>
	$<$<COMPILE_LANGUAGE:C>:-Wconversion>
	$<$<COMPILE_LANGUAGE:C>:-Wdisabled-optimization>
	$<$<COMPILE_LANGUAGE:C>:-Wdouble-promotion>
	$<$<COMPILE_LANGUAGE:C>:-Werror>
	$<$<COMPILE_LANGUAGE:C>:-Wextra>
	$<$<COMPILE_LANGUAGE:C>:-Wfloat-equal>
	$<$<COMPILE_LANGUAGE:C>:-Winit-self>
	$<$<COMPILE_LANGUAGE:C>:-Wmissing-declarations>
	$<$<COMPILE_LANGUAGE:C>:-Wmissing-include-dirs>
	$<$<COMPILE_LANGUAGE:C>:-Wshadow>
	$<$<COMPILE_LANGUAGE:C>:-Wstrict-prototypes>
	$<$<COMPILE_LANGUAGE:C>:-Wswitch-default>
	$<$<COMPILE_LANGUAGE:C>:-Wswitch-enum>
	$<$<COMPILE_LANGUAGE:C>:-Wundef>
	$<$<COMPILE_LANGUAGE:C>:-Wuninitialized>
	$<$<COMPILE_LANGUAGE:C>:-Wwrite-strings>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-fdata-sections>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-ffunction-sections>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-fno-pie>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-fno-plt>
	$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-funroll-loops>
	)
	if(SUPPORTS_GCC_FLAGS)
		target_compile_options(${target} PRIVATE
		$<$<COMPILE_LANGUAGE:C>:-fanalyzer>
		$<$<COMPILE_LANGUAGE:C>:-Warith-conversion>
		$<$<COMPILE_LANGUAGE:C>:-Wduplicated-branches>
		$<$<COMPILE_LANGUAGE:C>:-Wduplicated-cond>
		$<$<COMPILE_LANGUAGE:C>:-Wlogical-op>
		$<$<COMPILE_LANGUAGE:C>:-Wopenmp>
		$<$<COMPILE_LANGUAGE:C>:-Wuseless-cast>
		$<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>,$<BOOL:${HAS_GDB}>>:-ggdb3>
		$<$<AND:$<COMPILE_LANGUAGE:C>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-fprefetch-loop-arrays>
		)
	endif()
	if(NOT USE_ASAN)
		target_compile_options(${target} PRIVATE $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-fstack-protector-strong>)
	endif()
	if(PLATFORM_TYPE STREQUAL "GCC" AND USE_ASAN)
		target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
		target_link_options(${target} PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
	endif()
	target_link_options(${target} PRIVATE
	$<$<CONFIG:Release>:-Wl,-O3>
	$<$<CONFIG:RelWithDebInfo>:-Wl,-O2>
	$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-flto>
	$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-no-pie>
	$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-Wl,--as-needed>
	$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-Wl,--gc-sections>
	)
	if(WIN32)
		target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN)
	else()
		target_compile_definitions(${target} PRIVATE _GNU_SOURCE)
		target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C>:-pthread>)
	endif()
endif()
endfunction()

function(set_cuda_compiler_options target)
if(NOT USE_CUDA)
	return()
endif()
target_compile_definitions(${target} PRIVATE USE_CUDA)
target_compile_options(${target} PRIVATE
$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>,$<BOOL:${HAS_GDB}>>:-ggdb3>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-extra-device-vectorization>
)
if(PLATFORM_TYPE STREQUAL "MSVC")
	target_compile_options(${target} PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:${ARCH_LEVEL}>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Wall>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/analyze>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4514> # unreferenced inline function
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4668> # #if {undefined}
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4710> # not inlined info
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4711> # automatic inline info
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4820> # struct padding
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd5039> # potential throwing funcction in extern "C"
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd5045> # spectre mitigation
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/Od>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/MDd>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=/RTC1>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=/GL>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=/Ox>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-Xcompiler=/O2>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-Xcompiler=/Ob1>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=/Zi>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=/MD>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=/GA>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=/Gy>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=/Gw>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=/DNDEBUG>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=/jumptablerdata>
	)
elseif(PLATFORM_TYPE STREQUAL "GCC")
	target_compile_options(${target} PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=${ARCH_LEVEL}>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-pedantic>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wcast-align>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wcast-qual>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wconversion>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wdisabled-optimization>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wdouble-promotion>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wfloat-equal>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Winit-self>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wmissing-declarations>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wmissing-include-dirs>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wshadow>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wstrict-overflow=5>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wswitch-default>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wswitch-enum>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wundef>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wuninitialized>
	$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wwrite-strings>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=-O3>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-Xcompiler=-O2>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-Xcompiler=-fstack-protector-strong>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=-DNDEBUG>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=-fdata-sections>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=-ffunction-sections>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=-fno-pie>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=-fno-plt>
	$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=-funroll-loops>
	)
	if(SUPPORTS_GCC_FLAGS)
		target_compile_options(${target} PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fanalyzer>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Warith-conversion>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wduplicated-branches>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wduplicated-cond>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wlogical-op>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wuseless-cast>
		$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>,$<BOOL:${HAS_GDB}>>:-Xcompiler=-ggdb3>
		$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-Xcompiler=-fprefetch-loop-arrays>
		)
	endif()
endif()
endfunction()

add_executable(seqalign ${C_SOURCES})
if(USE_CUDA)
	target_sources(seqalign PRIVATE ${CUDA_SOURCES})
	set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

target_include_directories(seqalign PRIVATE ${CMAKE_SOURCE_DIR}/code/include)
if(USE_CUDA)
	target_include_directories(seqalign PRIVATE ${CMAKE_SOURCE_DIR}/code/cuda/include ${CMAKE_SOURCE_DIR}/code/cuda/c_binding)
endif()

set_c_compiler_options(seqalign)
set_cuda_compiler_options(seqalign)

if(LEGACY_HDF5 OR IS_CROSS OR PLATFORM_TYPE STREQUAL "MSYS2")
	target_include_directories(seqalign PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(seqalign PRIVATE ${HDF5_LIBRARIES})
	target_link_directories(seqalign PRIVATE ${HDF5_LIBRARY_DIRS})
else()
	target_link_libraries(seqalign PRIVATE ${LINK_LIBS})
endif()

if(WIN32)
	if(PLATFORM_TYPE STREQUAL "MSVC")
		target_link_libraries(seqalign PRIVATE Shlwapi)
	else()
		target_link_libraries(seqalign PRIVATE shlwapi)
	endif()
endif()

if(USE_CUDA)
	target_link_libraries(seqalign PRIVATE CUDA::cudart)
endif()

target_link_libraries(seqalign PRIVATE OpenMP::OpenMP_C)

function(install_runtime_dlls dll_list prefix)
foreach(dll ${dll_list})
	if(EXISTS "${prefix}/bin/${dll}")
		install(FILES "${prefix}/bin/${dll}" DESTINATION .)
	elseif(EXISTS "${prefix}/lib/${dll}")
		install(FILES "${prefix}/lib/${dll}" DESTINATION .)
	endif()
endforeach()
endfunction()

if(WIN32)
	if(PLATFORM_TYPE STREQUAL "MSVC")
		install(TARGETS seqalign 
		RUNTIME_DEPENDENCIES
		PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "vcruntime.*\\.dll"
		POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
		RUNTIME DESTINATION .
		)
	elseif(PLATFORM_TYPE STREQUAL "MSYS2")
		if(NOT DEFINED ENV{MSYSTEM_PREFIX})
			message(FATAL_ERROR "MSYSTEM_PREFIX not set. Cannot locate MSYS2 bin directory.")
		endif()
		set(MSYS2_DLLS
		libhdf5-310.dll libcrypto-3-x64.dll libwinpthread-1.dll libcurl-4.dll libsz-2.dll zlib1.dll
		libbrotlidec.dll libidn2-0.dll libnghttp2-14.dll libnghttp3-9.dll libpsl-5.dll libssh2-1.dll
		libbrotlicommon.dll libzstd.dll libssl-3-x64.dll libintl-8.dll libiconv-2.dll libunistring-5.dll
		libgomp-1.dll libgcc_s_seh-1.dll libngtcp2-16.dll libngtcp2_crypto_ossl-0.dll
		)
		install_runtime_dlls("${MSYS2_DLLS}" "$ENV{MSYSTEM_PREFIX}")
	elseif(IS_CROSS)        
		set(CROSS_DLLS libhdf5.dll libssp-0.dll libwinpthread-1.dll libsz.dll zlib1.dll libgomp-1.dll libgcc_s_seh-1.dll)
		install_runtime_dlls("${CROSS_DLLS}" "${MINGW_PREFIX}")
	endif()
endif()

if(USE_BIN)
	set_target_properties(seqalign PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
	if(PLATFORM_TYPE STREQUAL "MSVC")
		add_custom_command(TARGET seqalign POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_FILE:seqalign>
		"${CMAKE_SOURCE_DIR}/bin/"
		)
		install_runtime_dlls("$<TARGET_FILE_DIR:seqalign>" "${CMAKE_SOURCE_DIR}/bin")
	endif()
endif()

install(TARGETS seqalign RUNTIME DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" "${CMAKE_SOURCE_DIR}/README.md" DESTINATION .)

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(compile_commands)
