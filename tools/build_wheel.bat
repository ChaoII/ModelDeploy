@echo off
REM ============================================================================
REM  ModelDeploy - Python wheel build entry (python -m build)
REM
REM  Edit the switches in "BUILD OPTIONS" below, then run this batch from the
REM  project ROOT (where pyproject.toml lives). It sets the env vars that
REM  scikit-build-core maps to CMake (see pyproject.toml [tool.scikit-build]).
REM
REM  Output: dist\*.whl
REM ============================================================================
setlocal

REM ---- Python / toolchain ----
set "PYTHON=python"

REM ---- Detect & load the MSVC x64 environment (Needed by Ninja on Windows).
REM       If you already run this from a "x64 Native Tools Command Prompt",
REM       skip by setting USE_EXISTING_VC=1 . ----------------------------------
set "USE_EXISTING_VC=0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if "%USE_EXISTING_VC%"=="0" (
    if exist "%VSWHERE%" (
        for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCBASE=%%i"
        if defined VCBASE (
            echo [VC] using %VCBASE%
            call "%VCBASE%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
        ) else (
            echo [WARN] Visual Studio C++ tools not found. Open an "x64 Native Tools Command Prompt" instead.
        )
    ) else (
        echo [WARN] vswhere not found. Open an "x64 Native Tools Command Prompt" instead.
    )
)

REM ============================================================================
REM  BUILD OPTIONS  (edit these)
REM ============================================================================
REM Backend / accelerator switches (mapped to CMake via env in pyproject.toml):
set "WITH_GPU=OFF"            REM ON=enable CUDA, OFF=CPU only
set "ENABLE_ORT=ON"           REM OnnxRuntime backend
set "ENABLE_MNN=OFF"          REM MNN backend
set "ENABLE_TRT=OFF"          REM TensorRT backend (requires WITH_GPU=ON)
set "TRT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.9.0.34"
set "BUILD_ENCRYPTION=OFF"    REM model encryption (needs mbedTLS submodule; run git submodule update --init --recursive first)

REM ---- Extra CMake args (valuable for vars NOT mapped via env in pyproject). ----
set "SKBUILD_CMAKE_ARGS="

REM ============================================================================
REM  Build
REM ============================================================================
echo.
echo [Build] WITH_GPU=%WITH_GPU% ENABLE_ORT=%ENABLE_ORT% ENABLE_MNN=%ENABLE_MNN% ENABLE_TRT=%ENABLE_TRT%
echo [Build] extra CMake args: %SKBUILD_CMAKE_ARGS%
echo.

"%PYTHON%" -c "import scikit_build_core" >nul 2>&1
if errorlevel 1 (
    echo [INFO] installing build requirements...
    "%PYTHON%" -m pip install --upgrade pip build
)

"%PYTHON%" -m build
if errorlevel 1 (
    echo [ERROR] wheel build failed.
    exit /b 1
)

echo.
echo [OK] Built wheel. Check dist\ for the output artifact(s).
endlocal
