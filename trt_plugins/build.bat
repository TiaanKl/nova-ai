@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

echo === Cleaning build directory ===
if exist build (
    rmdir /s /q build
)

echo === Creating build directory ===
mkdir build
cd build

set "CMAKE_EXE="
for /f "delims=" %%I in ('where cmake 2^>nul') do (
    if not defined CMAKE_EXE set "CMAKE_EXE=%%I"
)

if not defined CMAKE_EXE if exist "C:\Program Files\CMake\bin\cmake.exe" set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"
if not defined CMAKE_EXE if exist "C:\Program Files (x86)\CMake\bin\cmake.exe" set "CMAKE_EXE=C:\Program Files (x86)\CMake\bin\cmake.exe"
if not defined CMAKE_EXE if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not defined CMAKE_EXE if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" set "CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

if not defined CMAKE_EXE (
    echo cmake.exe was not found. Please install CMake or run this from a Visual Studio Developer Prompt.
    popd
    exit /b 1
)

echo === Configuring CMake (Debug) ===
"%CMAKE_EXE%" .. -G "Ninja" ^
    -DTENSORRT_ROOT="C:/SDKs/tensorrt" ^
    -DCMAKE_BUILD_TYPE=Debug

if %errorlevel% neq 0 (
    echo CMake configuration failed.
    popd
    exit /b 1
)

echo === Building with Ninja ===
"%CMAKE_EXE%" --build . --parallel

if %errorlevel% neq 0 (
    echo Build failed.
    popd
    exit /b 1
)

echo === Build completed successfully ===
popd
endlocal
