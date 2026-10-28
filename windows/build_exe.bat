@echo off
rem ============================================================
rem  Maestro Windows exe builder (Iris Xe iGPU via DirectML)
rem  Author: Vaibhav (sole author; final authority on this project)
rem  License: GPL-3.0-only <https://www.gnu.org/licenses/gpl-3.0.html>
rem ============================================================
echo Installing DirectML + build tools (one time, few minutes)...
python -m pip install --quiet torch-directml mido numpy rich tqdm pyinstaller music21
if errorlevel 1 (
    echo pip install failed - is Python 3.10-3.12 on PATH?
    pause
    exit /b 1
)
echo Building MaestroComposer.exe (onefile, this takes a few minutes)...
python -m PyInstaller --onefile --name MaestroComposer --paths .. --paths . ^
  --hidden-import musicbot --hidden-import musicbot.tokenizer --hidden-import musicbot.model ^
  --hidden-import musicbot.generate --hidden-import musicbot.rl --hidden-import musicbot.cli ^
  --collect-submodules torch_directml ^
  compose_windows.py
if errorlevel 1 (
    echo build failed
    pause
    exit /b 1
)
echo.
echo Done. Run:  dist\MaestroComposer.exe
pause
