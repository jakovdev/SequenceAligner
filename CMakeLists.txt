cmake_minimum_required(VERSION 3.30)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
project(SequenceAligner C)

set(ARCH_LEVEL
    "native"
    CACHE STRING "Architecture level")

set(IS_MINGW FALSE)
set(IS_MSYS2 FALSE)
set(HDF5_TARGET hdf5::hdf5)
if("$ENV{MSYSTEM}" STREQUAL "UCRT64")
  set(IS_MSYS2 TRUE)
  set(HDF5_USE_STATIC_LIBRARIES TRUE)
  set(HDF5_TARGET hdf5-static -static winpthread -l:libgomp.a)
elseif(MINGW)
  set(IS_MINGW TRUE)
  string(LENGTH "${CMAKE_INSTALL_LIBDIR}" ____unused)
  string(LENGTH "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}" ____unused)
  list(GET CMAKE_FIND_ROOT_PATH 0 DLL_PREFIX)
  set(libaec_USE_STATIC_LIBS TRUE)
  find_package(libaec REQUIRED)
  add_library(libaec::aec-shared ALIAS libaec::aec-static)
  add_library(libaec::sz-shared ALIAS libaec::sz-static)
  set(ZLIB_USE_STATIC_LIBS ON)
  set(HDF5_USE_STATIC_LIBRARIES TRUE)
  set(HDF5_DIR "${DLL_PREFIX}/cmake/hdf5")
  set(HDF5_TARGET hdf5-static -static winpthread -l:libgomp.a)
endif()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED COMPONENTS C)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.c")
add_executable(seqalign ${SOURCES})

target_include_directories(
  seqalign PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party/clix)

if(NOT IS_MINGW)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(CUDA_KERNELS "src/generated/kernels.fatbin")
    set(CUDA_KERNELS_PATH "${CMAKE_SOURCE_DIR}/${CUDA_KERNELS}")
    set(CUDA_KERNELS_SRC "${CMAKE_SOURCE_DIR}/src/bio/kernels.cu")
    set(CUDA_KERNELS_DEPFILE "${CUDA_KERNELS_PATH}.d")

    set(NVCC_COMMAND
        "${CUDAToolkit_NVCC_EXECUTABLE}" "-I${CMAKE_SOURCE_DIR}/src"
        "-arch=all" "-fatbin" "${CUDA_KERNELS_SRC}" "-o" "${CUDA_KERNELS_PATH}")

    add_custom_command(
      OUTPUT "${CUDA_KERNELS_PATH}"
      COMMAND ${NVCC_COMMAND} -MD -MF "${CUDA_KERNELS_DEPFILE}" -MT
              "${CUDA_KERNELS_PATH}"
      DEPENDS "${CUDA_KERNELS_SRC}"
      DEPFILE "${CUDA_KERNELS_DEPFILE}"
      COMMENT "Building CUDA kernels ${CUDA_KERNELS}")

    add_custom_target(kernels DEPENDS "${CUDA_KERNELS_PATH}")
    add_dependencies(seqalign kernels)
  else()
    message(STATUS "Cuda Toolkit not found, using cached kernels")
  endif()
endif()

target_compile_definitions(seqalign PRIVATE $<$<BOOL:${IS_MINGW}>:MINGW_LIBSZ>)
target_compile_options(seqalign PRIVATE -Wall -Wextra -flto=auto -fno-plt
                                        -march=${ARCH_LEVEL} -funroll-loops)
target_link_options(seqalign PRIVATE -flto=auto)
target_link_libraries(seqalign PRIVATE ${HDF5_TARGET})
target_link_libraries(seqalign PRIVATE OpenMP::OpenMP_C)

install(TARGETS seqalign RUNTIME DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" "${CMAKE_SOURCE_DIR}/README.md"
        DESTINATION .)

set(CPACK_PACKAGE_NAME "SequenceAligner")
set(CPACK_PACKAGE_VENDOR "Jakov Dragičević")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "High-performance all-vs-all pairwise sequence alignment tool")
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
