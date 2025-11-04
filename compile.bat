@echo off
REM -------------------------------------------
REM Auto-setup for Visual Studio 2022 + Ninja
REM -------------------------------------------

REM Change to the directory of the script
cd /d "%~dp0"

REM Call VS 2022 environment setup for x64
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
IF ERRORLEVEL 1 (
    echo Failed to initialize Visual Studio environment!
    exit /b 1
)

REM Create build directory if it doesn't exist
if not exist out\build-relwithdebinfo mkdir out\build-relwithdebinfo

REM Change into build directory
cd out\build-relwithdebinfo

REM Run CMake with Ninja generator
cmake ..\.. -G Ninja ^
    -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
    -DCMAKE_C_COMPILER="cl.exe" ^
    -DCMAKE_CXX_COMPILER="cl.exe" ^
    -DCMAKE_MAKE_PROGRAM="C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"

REM Build the project
ninja

pause
