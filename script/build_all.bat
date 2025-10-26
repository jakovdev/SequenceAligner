@echo off
setlocal enabledelayedexpansion

set BUILD_DIR=build
mkdir "%BUILD_DIR%" 2>nul

where nvcc >nul 2>&1
if %errorlevel% equ 0 (
    set USE_CUDA=ON
) else (
    set USE_CUDA=OFF
)

set ARCHS=SSE2 SSE4.2 AVX AVX2 AVX512 AVX10.1

for %%a in (%ARCHS%) do (
    echo.
    set build_subdir=%BUILD_DIR%/%%a-windows
    if "!USE_CUDA!"=="yes" (
        set build_subdir=%build_subdir%-cuda
    )
    rmdir /s /q "!build_subdir!" 2>nul
    mkdir "!build_subdir!"
    cmake -S . -B "!build_subdir!" -G Ninja --toolchain "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release -DARCH_LEVEL=%%a -DUSE_CUDA="!USE_CUDA!"
    if errorlevel 1 (
        echo [ERROR] CMake configure failed for "!build_subdir!"
        exit /b 1
    )
    cmake --build "!build_subdir!" --config Release -- -j %NUMBER_OF_PROCESSORS%
    if errorlevel 1 (
        echo [ERROR] CMake build failed for "!build_subdir!"
        exit /b 1
    )
    cd "!build_subdir!"
    cpack
    if errorlevel 1 (
        echo [ERROR] CPack failed for "!build_subdir!"
        exit /b 1
    )
    cd ..\..
)

endlocal
exit /b 0
