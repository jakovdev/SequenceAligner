cmake_minimum_required(VERSION 3.30)
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
project(SequenceAligner C CXX)

set(ARCH_LEVEL "native" CACHE STRING "Architecture level")

set(IS_MINGW FALSE)
set(IS_MSYS2 FALSE)

if(MINGW_PREFIX)
	set(IS_MINGW TRUE)
	set(DLL_PREFIX "${MINGW_PREFIX}")
	set(DLL_LIST libhdf5.dll libwinpthread-1.dll libsz.dll libgomp-1.dll libgcc_s_seh-1.dll libstdc++-6.dll libssp-0.dll zlib1.dll)
elseif(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "GNU")
	set(IS_MSYS2 TRUE)
	set(DLL_PREFIX "$ENV{MSYSTEM_PREFIX}")
	file(GLOB HDF5_DLL "${DLL_PREFIX}/bin/libhdf5-[0-9]*.dll")
	string(REGEX REPLACE ".+/(libhdf5-[0-9]+\\.dll)" "\\1" HDF5_DLL "${HDF5_DLL}")
	set(DLL_LIST ${HDF5_DLL}
	libwinpthread-1.dll libsz-2.dll libgomp-1.dll libgcc_s_seh-1.dll libstdc++-6.dll libz-ng2.dll
	libaws-c-io.dll libaws-c-s3.dll libaws-c-auth.dll libaws-c-http.dll libaws-c-common.dll
	libaws-c-cal.dll libaws-c-sdkutils.dll libaws-c-compression.dll libaws-checksums.dll
	)
endif()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(USE_CUDA FALSE)
if(NOT IS_MINGW)
	find_package(CUDAToolkit)
	if(CUDAToolkit_FOUND)
		if(NOT IS_MSYS2)
			enable_language(CUDA)
			set(CMAKE_CUDA_STANDARD 20)
			set(CMAKE_CUDA_STANDARD_REQUIRED ON)
		else()
			find_program(GENDEF_EXECUTABLE gendef REQUIRED)
			find_program(DLLTOOL_EXECUTABLE dlltool REQUIRED)
			file(GLOB CUDA_MSYS2_CUDART_DLLS LIST_DIRECTORIES FALSE "$ENV{CUDA_PATH}/bin/x64/cudart64_*.dll")
			if(NOT CUDA_MSYS2_CUDART_DLLS)
				message(FATAL_ERROR "Could not find cudart64_*.dll under $ENV{CUDA_PATH}/bin/x64")
			endif()
			list(SORT CUDA_MSYS2_CUDART_DLLS)
			list(GET CUDA_MSYS2_CUDART_DLLS 0 CUDA_MSYS2_CUDART_DLL)
			get_filename_component(CUDA_MSYS2_CUDART_DLL_NAME "${CUDA_MSYS2_CUDART_DLL}" NAME)
			get_filename_component(CUDA_MSYS2_CUDART_DLL_STEM "${CUDA_MSYS2_CUDART_DLL_NAME}" NAME_WE)
			set(CUDA_MSYS2_RUNTIME_DIR "${CMAKE_BINARY_DIR}/msys2_cuda")
			set(CUDA_MSYS2_QUIET_RUNNER "${CUDA_MSYS2_RUNTIME_DIR}/run_quiet.cmake")
			set(CUDA_MSYS2_CUDART_IMPORT_LIB "${CUDA_MSYS2_RUNTIME_DIR}/libcudart.dll.a")
			set(CUDA_MSYS2_CUDART_DLL_COPY "${CUDA_MSYS2_RUNTIME_DIR}/${CUDA_MSYS2_CUDART_DLL_NAME}")
			file(MAKE_DIRECTORY "${CUDA_MSYS2_RUNTIME_DIR}")
			execute_process(
				COMMAND "${GENDEF_EXECUTABLE}" "${CUDA_MSYS2_CUDART_DLL}"
				WORKING_DIRECTORY "${CUDA_MSYS2_RUNTIME_DIR}"
				RESULT_VARIABLE CUDA_MSYS2_GENDEF_RESULT
				OUTPUT_QUIET
				ERROR_QUIET
			)
			if(NOT CUDA_MSYS2_GENDEF_RESULT EQUAL 0)
				execute_process(
					COMMAND "${GENDEF_EXECUTABLE}" "${CUDA_MSYS2_CUDART_DLL}"
					WORKING_DIRECTORY "${CUDA_MSYS2_RUNTIME_DIR}"
					RESULT_VARIABLE CUDA_MSYS2_GENDEF_RETRY_RESULT
					OUTPUT_VARIABLE CUDA_MSYS2_GENDEF_STDOUT
					ERROR_VARIABLE CUDA_MSYS2_GENDEF_STDERR
				)
				message(FATAL_ERROR "Failed to generate cudart def from ${CUDA_MSYS2_CUDART_DLL}:\n${CUDA_MSYS2_GENDEF_STDOUT}${CUDA_MSYS2_GENDEF_STDERR}")
			endif()
			execute_process(
				COMMAND "${DLLTOOL_EXECUTABLE}" -d "${CUDA_MSYS2_CUDART_DLL_STEM}.def" -l "${CUDA_MSYS2_CUDART_IMPORT_LIB}" -D "${CUDA_MSYS2_CUDART_DLL_NAME}"
				WORKING_DIRECTORY "${CUDA_MSYS2_RUNTIME_DIR}"
				RESULT_VARIABLE CUDA_MSYS2_DLLTOOL_RESULT
				OUTPUT_QUIET
				ERROR_QUIET
			)
			if(NOT CUDA_MSYS2_DLLTOOL_RESULT EQUAL 0)
				execute_process(
					COMMAND "${DLLTOOL_EXECUTABLE}" -d "${CUDA_MSYS2_CUDART_DLL_STEM}.def" -l "${CUDA_MSYS2_CUDART_IMPORT_LIB}" -D "${CUDA_MSYS2_CUDART_DLL_NAME}"
					WORKING_DIRECTORY "${CUDA_MSYS2_RUNTIME_DIR}"
					RESULT_VARIABLE CUDA_MSYS2_DLLTOOL_RETRY_RESULT
					OUTPUT_VARIABLE CUDA_MSYS2_DLLTOOL_STDOUT
					ERROR_VARIABLE CUDA_MSYS2_DLLTOOL_STDERR
				)
				message(FATAL_ERROR "Failed to create cudart import library at ${CUDA_MSYS2_CUDART_IMPORT_LIB}:\n${CUDA_MSYS2_DLLTOOL_STDOUT}${CUDA_MSYS2_DLLTOOL_STDERR}")
			endif()
			file(COPY_FILE "${CUDA_MSYS2_CUDART_DLL}" "${CUDA_MSYS2_CUDART_DLL_COPY}" ONLY_IF_DIFFERENT)
			file(WRITE "${CUDA_MSYS2_QUIET_RUNNER}" [=[
if(NOT DEFINED ARGN_LIST)
	message(FATAL_ERROR "run_quiet.cmake requires ARGN_LIST")
endif()
execute_process(
	COMMAND ${ARGN_LIST}
	WORKING_DIRECTORY "${WORKING_DIRECTORY}"
	RESULT_VARIABLE RESULT
	OUTPUT_QUIET
)
if(NOT RESULT EQUAL 0)
	execute_process(
		COMMAND ${ARGN_LIST}
		WORKING_DIRECTORY "${WORKING_DIRECTORY}"
		RESULT_VARIABLE RETRY_RESULT
		OUTPUT_VARIABLE STDOUT
		ERROR_VARIABLE STDERR
	)
	message(FATAL_ERROR "Command failed with code ${RESULT}: ${ARGN_LIST}\n${STDOUT}${STDERR}")
endif()
]=])
			find_program(OBJCOPY_EXECUTABLE objcopy)
			set(CUDA_KERNEL_OBJ "${CMAKE_BINARY_DIR}/kernels.cu.obj")
			set(CUDA_KERNEL_SRC "${CMAKE_SOURCE_DIR}/src/bio/kernels.cu")
			set(CUDA_MSYS2_NVCC_CMD "nvcc;-forward-unknown-to-host-compiler;-Xcompiler=/GS-,/sdl-,/MT,/Zc:threadSafeInit-;-I;${CMAKE_SOURCE_DIR}/include;-O3;-DNDEBUG;-std=c++20;--generate-code=arch=compute_75,code=[compute_75,sm_75];-cudart=static;-x;cu;-c;${CUDA_KERNEL_SRC};-o;${CUDA_KERNEL_OBJ}")
			set(CUDA_MSYS2_POST_NVCC_CMD)
			if(OBJCOPY_EXECUTABLE)
				set(CUDA_MSYS2_POST_NVCC_CMD COMMAND "${OBJCOPY_EXECUTABLE}" --remove-section .drectve "${CUDA_KERNEL_OBJ}")
			endif()
			add_custom_command(
				OUTPUT "${CUDA_KERNEL_OBJ}"
				COMMAND "${CMAKE_COMMAND}" "-DWORKING_DIRECTORY:PATH=${CMAKE_BINARY_DIR}" "-DARGN_LIST:STRING=${CUDA_MSYS2_NVCC_CMD}" -P "${CUDA_MSYS2_QUIET_RUNNER}"
				${CUDA_MSYS2_POST_NVCC_CMD}
				DEPENDS "${CUDA_KERNEL_SRC}"
				COMMENT "Building CUDA object ${CUDA_KERNEL_OBJ}"
			)
			add_custom_target(kernels DEPENDS "${CUDA_KERNEL_OBJ}")
		endif()
		set(USE_CUDA TRUE)
	endif()
endif()

if(IS_MSYS2 OR IS_MINGW)
	find_package(PkgConfig REQUIRED)
	pkg_check_modules(HDF5 REQUIRED hdf5)
else()
	find_package(HDF5 REQUIRED COMPONENTS C)
endif()

find_package(OpenMP REQUIRED COMPONENTS C)

file(GLOB_RECURSE C_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.c")
file(GLOB_RECURSE CXX_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
if(USE_CUDA AND IS_MSYS2)
	add_executable(seqalign ${C_SOURCES} ${CXX_SOURCES} ${CUDA_KERNEL_OBJ})
	add_dependencies(seqalign kernels)
	target_include_directories(seqalign PRIVATE "$ENV{CUDA_PATH}/include")
	target_link_libraries(seqalign PRIVATE "${CUDA_MSYS2_CUDART_IMPORT_LIB}")
else()
	add_executable(seqalign ${C_SOURCES} ${CXX_SOURCES})
endif()
set_target_properties(seqalign PROPERTIES LINKER_LANGUAGE C)

if(USE_CUDA AND NOT IS_MSYS2)
	file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cu")
	target_sources(seqalign PRIVATE ${CUDA_SOURCES})
endif()

target_include_directories(seqalign PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/third_party/clix)

if(IS_MSYS2 OR IS_MINGW)
	target_include_directories(seqalign PRIVATE "${CMAKE_SOURCE_DIR}/third_party/c11threads")
endif()

target_compile_definitions(seqalign PRIVATE _GNU_SOURCE $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:${USE_CUDA}>>:USE_CUDA>)
target_compile_options(seqalign PRIVATE
$<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-march=${ARCH_LEVEL};-funroll-loops;-fwrapv;-fno-plt;-fvisibility=hidden;-malign-data=cacheline;-flto>
$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-march=${ARCH_LEVEL};-funroll-loops;-fwrapv;-fno-plt;-fvisibility=hidden;-malign-data=cacheline;-flto;-fno-exceptions;-fno-rtti>
$<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-Og>
$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Og>
$<$<COMPILE_LANGUAGE:CUDA>:-cudart=static>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)
target_link_options(seqalign PRIVATE -flto=auto)

if(IS_MSYS2 OR IS_MINGW)
	target_include_directories(seqalign PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(seqalign PRIVATE ${HDF5_LIBRARIES})
	target_link_directories(seqalign PRIVATE ${HDF5_LIBRARY_DIRS})
else()
	target_link_libraries(seqalign PRIVATE hdf5::hdf5)
endif()

if(USE_CUDA AND NOT IS_MSYS2)
	target_link_libraries(seqalign PRIVATE CUDA::cudart_static)
endif()

target_link_libraries(seqalign PRIVATE OpenMP::OpenMP_C)

if(IS_MSYS2 OR IS_MINGW)
	foreach(dll ${DLL_LIST})
		if(EXISTS "${DLL_PREFIX}/bin/${dll}")
			install(FILES "${DLL_PREFIX}/bin/${dll}" DESTINATION .)
			file(COPY "${DLL_PREFIX}/bin/${dll}" DESTINATION "${CMAKE_BINARY_DIR}")
		elseif(EXISTS "${DLL_PREFIX}/lib/${dll}")
			install(FILES "${DLL_PREFIX}/lib/${dll}" DESTINATION .)
			file(COPY "${DLL_PREFIX}/lib/${dll}" DESTINATION "${CMAKE_BINARY_DIR}")
		else()
			message(WARNING "Could not find runtime DLL ${dll} in expected locations under prefix ${DLL_PREFIX}")
		endif()
	endforeach()
endif()

if(USE_CUDA AND IS_MSYS2)
	install(FILES "${CUDA_MSYS2_CUDART_DLL_COPY}" DESTINATION .)
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

if(IS_MSYS2 OR IS_MINGW)
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-windows-${ARCH_LEVEL}")
	set(CPACK_GENERATOR "ZIP")
else()
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-linux-${ARCH_LEVEL}")
	set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)
