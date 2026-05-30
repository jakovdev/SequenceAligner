cmake_minimum_required(VERSION 3.30)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
project(SequenceAligner C CXX)

set(ARCH_LEVEL
    "native"
    CACHE STRING "Architecture level")

set(IS_MINGW FALSE)
set(IS_MSYS2 FALSE)

if("$ENV{MSYSTEM}" STREQUAL "UCRT64")
  set(IS_MSYS2 TRUE)
  set(DLL_PREFIX "$ENV{MSYSTEM_PREFIX}")
  file(GLOB HDF5_DLL "${DLL_PREFIX}/bin/libhdf5-[0-9]*.dll")
  if(NOT HDF5_DLL)
    message(FATAL_ERROR "Could not find hdf5 under ${DLL_PREFIX}/bin")
  endif()
  string(REGEX REPLACE ".+/(libhdf5-[0-9]+\\.dll)" "\\1" HDF5_DLL "${HDF5_DLL}")
  set(DLL_LIST
      ${HDF5_DLL}
      libwinpthread-1.dll
      libsz-2.dll
      libgomp-1.dll
      libgcc_s_seh-1.dll
      libstdc++-6.dll
      libz-ng2.dll
      libaws-c-io.dll
      libaws-c-s3.dll
      libaws-c-auth.dll
      libaws-c-http.dll
      libaws-c-common.dll
      libaws-c-cal.dll
      libaws-c-sdkutils.dll
      libaws-c-compression.dll
      libaws-checksums.dll)
elseif(MINGW)
  set(IS_MINGW TRUE)
  string(LENGTH "${CMAKE_INSTALL_LIBDIR}" ____unused)
  list(GET CMAKE_FIND_ROOT_PATH 0 DLL_PREFIX)
  set(DLL_LIST
      libhdf5.dll
      libwinpthread-1.dll
      libsz.dll
      libgomp-1.dll
      libgcc_s_seh-1.dll
      libstdc++-6.dll
      libssp-0.dll
      zlib1.dll)
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
      find_program(GENDEF gendef REQUIRED)
      find_program(DLLTOOL dlltool REQUIRED)
      find_program(OBJCOPY objcopy REQUIRED)
      file(
        GLOB CUDART_SYS_DLL
        LIST_DIRECTORIES FALSE
        "$ENV{CUDA_PATH}/bin/x64/cudart64_*.dll")
      if(NOT CUDART_SYS_DLL)
        message(
          FATAL_ERROR
            "Could not find cudart64_*.dll under $ENV{CUDA_PATH}/bin/x64")
      endif()

      set(MSYS2_CUDA "${CMAKE_BINARY_DIR}/msys2_cuda")
      file(MAKE_DIRECTORY "${MSYS2_CUDA}")
      get_filename_component(CUDART_DLL_NM "${CUDART_SYS_DLL}" NAME)
      get_filename_component(CUDART_DLL_WE "${CUDART_DLL_NM}" NAME_WE)
      set(CUDART_DLL "${MSYS2_CUDA}/${CUDART_DLL_NM}")
      file(COPY_FILE "${CUDART_SYS_DLL}" "${CUDART_DLL}" ONLY_IF_DIFFERENT)
      file(COPY_FILE "${CUDART_SYS_DLL}" "${CUDART_DLL_NM}" ONLY_IF_DIFFERENT)

      set(CUDART_DEF "${MSYS2_CUDA}/${CUDART_DLL_WE}.def")
      set(CUDART_LIB "${MSYS2_CUDA}/libcudart.dll.a")
      add_custom_command(
        OUTPUT "${CUDART_DEF}"
        COMMAND "${GENDEF}" "${CUDART_DLL}"
        WORKING_DIRECTORY "${MSYS2_CUDA}"
        DEPENDS "${CUDART_DLL}"
        COMMENT "Generating ${CUDART_DLL_WE}.def from ${CUDART_DLL_NM}")
      add_custom_command(
        OUTPUT "${CUDART_LIB}"
        COMMAND "${DLLTOOL}" -d "${CUDART_DEF}" -l "${CUDART_LIB}" -D
                "${CUDART_DLL_NM}"
        WORKING_DIRECTORY "${MSYS2_CUDA}"
        DEPENDS "${CUDART_DEF}"
        COMMENT "Creating import library libcudart.dll.a")
      add_custom_target(cudart_lib DEPENDS "${CUDART_LIB}")

      set(CUDA_KERNEL_OBJ "${MSYS2_CUDA}/kernels.cu.obj")
      set(CUDA_KERNEL_SRC "${CMAKE_SOURCE_DIR}/src/bio/kernels.cu")
      set(NVCC_COMMAND
          "nvcc;-Xcompiler=/GS-,/GR-,/Zl,/Zc:threadSafeInit-;-I;${CMAKE_SOURCE_DIR}/src;-O3;-DNDEBUG;-std=c++20;--generate-code=arch=compute_75,code=[compute_75,sm_75];-cudart=static;-x;cu;-c;${CUDA_KERNEL_SRC};-o;${CUDA_KERNEL_OBJ}"
      )
      add_custom_command(
        OUTPUT "${CUDA_KERNEL_OBJ}"
        COMMAND ${NVCC_COMMAND}
        COMMAND "${OBJCOPY}" --remove-section .drectve "${CUDA_KERNEL_OBJ}"
        DEPENDS "${CUDA_KERNEL_SRC}"
        COMMENT "Building CUDA kernel object")
      add_custom_target(kernels DEPENDS "${CUDA_KERNEL_OBJ}")
      add_dependencies(kernels cudart_lib)
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
  target_link_libraries(seqalign PRIVATE "${CUDART_LIB}")
else()
  add_executable(seqalign ${C_SOURCES} ${CXX_SOURCES})
endif()
set_target_properties(seqalign PROPERTIES LINKER_LANGUAGE C)

if(USE_CUDA AND NOT IS_MSYS2)
  file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_SOURCE_DIR}/src/*.cu")
  target_sources(seqalign PRIVATE ${CUDA_SOURCES})
endif()

target_include_directories(
  seqalign PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party/clix)

if(IS_MSYS2 OR IS_MINGW)
  target_include_directories(
    seqalign PRIVATE "${CMAKE_SOURCE_DIR}/third_party/c11threads")
endif()

target_compile_definitions(
  seqalign PRIVATE $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:${USE_CUDA}>>:USE_CUDA>)
target_compile_options(
  seqalign
  PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-march=${ARCH_LEVEL};-funroll-loops;-fwrapv;-fno-plt;-fvisibility=hidden;-malign-data=cacheline;-flto>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-march=${ARCH_LEVEL};-funroll-loops;-fwrapv;-fno-plt;-fvisibility=hidden;-malign-data=cacheline;-flto;-fno-exceptions;-fno-rtti>
    $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-Og>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-Og>
    $<$<COMPILE_LANGUAGE:CUDA>:-cudart=static>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
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
      message(
        WARNING
          "Could not find runtime DLL ${dll} in expected locations under prefix ${DLL_PREFIX}"
      )
    endif()
  endforeach()
endif()

if(USE_CUDA AND IS_MSYS2)
  install(FILES "${CUDART_DLL}" DESTINATION .)
endif()

install(TARGETS seqalign RUNTIME DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" "${CMAKE_SOURCE_DIR}/README.md"
        DESTINATION .)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(COMPILE_COMMANDS_SOURCE "${CMAKE_BINARY_DIR}/compile_commands.json")
  set(COMPILE_COMMANDS_ROOT "${CMAKE_SOURCE_DIR}/compile_commands.json")
  if(EXISTS "${COMPILE_COMMANDS_ROOT}" OR IS_SYMLINK "${COMPILE_COMMANDS_ROOT}")
    file(REMOVE "${COMPILE_COMMANDS_ROOT}")
  endif()
  file(CREATE_LINK "${COMPILE_COMMANDS_SOURCE}" "${COMPILE_COMMANDS_ROOT}"
       SYMBOLIC COPY_ON_ERROR RESULT COMPILE_COMMANDS_LINK_RESULT)
  if(NOT COMPILE_COMMANDS_LINK_RESULT STREQUAL "0"
     AND NOT EXISTS "${COMPILE_COMMANDS_ROOT}")
    message(
      WARNING
        "Failed to link compile_commands.json to project root: ${COMPILE_COMMANDS_LINK_RESULT}"
    )
  endif()
endif()

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
