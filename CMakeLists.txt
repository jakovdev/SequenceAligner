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
  set(HDF5_TARGET hdf5-static)
  set(DLL_PREFIX "$ENV{MSYSTEM_PREFIX}")
  set(DLL_LIST libwinpthread-1.dll libgomp-1.dll)
elseif(MINGW)
  set(IS_MINGW TRUE)
  string(LENGTH "${CMAKE_INSTALL_LIBDIR}" ____unused)
  string(LENGTH "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}" ____unused)
  list(GET CMAKE_FIND_ROOT_PATH 0 DLL_PREFIX)
  set(DLL_LIST
      libwinpthread-1.dll
      libssp-0.dll
      libgomp-1.dll
      libgcc_s_seh-1.dll
      libhdf5.dll
      libsz.dll
      zlib1.dll)
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

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(CUDA_KERNELS "src/generated/kernels.fatbin")
  set(CUDA_KERNELS_PATH "${CMAKE_SOURCE_DIR}/${CUDA_KERNELS}")
  set(CUDA_KERNELS_SRC "${CMAKE_SOURCE_DIR}/src/bio/kernels.cu")
  set(CUDA_KERNELS_DEPFILE "${CUDA_KERNELS_PATH}.d")

  set(NVCC_COMMAND
      "${CUDAToolkit_NVCC_EXECUTABLE}" "-I${CMAKE_SOURCE_DIR}/src" "-arch=all"
      "-fatbin" "${CUDA_KERNELS_SRC}" "-o" "${CUDA_KERNELS_PATH}")

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

target_compile_options(seqalign PRIVATE -Wall -Wextra -flto=auto -fno-plt
                                        -march=${ARCH_LEVEL} -funroll-loops)
target_link_options(seqalign PRIVATE -flto=auto)

target_link_libraries(seqalign PRIVATE ${HDF5_TARGET})
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
      message(WARNING "Could not find runtime DLL ${dll} under ${DLL_PREFIX}")
    endif()
  endforeach()
endif()

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
