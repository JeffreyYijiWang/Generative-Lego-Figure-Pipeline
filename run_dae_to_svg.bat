@echo off
REM === paths ===
set "BLENDER_EXE=C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
set "SCRIPT=%~dp0dae_to_svg_folder.py"
set "INPUT_DIR=%~dp0Input"
set "OUT_DIR=%~dp0Output"

if not exist "%INPUT_DIR%" (
  echo [ERROR] Input folder not found: "%INPUT_DIR%"
  exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo [INFO] Launching Blender once to process all .dae in "%INPUT_DIR%"
"%BLENDER_EXE%" -b -noaudio -P "%SCRIPT%" -- ^
  --inputdir "%INPUT_DIR%" ^
  --outdir "%OUT_DIR%" ^
  --pngsuffix "_render.png"

echo [DONE] Check outputs in "%OUT_DIR%"
