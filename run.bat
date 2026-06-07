@echo off
REM Transformer-GAT MedRec launcher for Windows (calls run.ps1).
REM Clinical decision support only — physician review required.
REM
REM   run.bat
REM   run.bat ui
REM   run.bat stop
REM   run.bat all

setlocal
cd /d "%~dp0"

where powershell >nul 2>&1
if errorlevel 1 (
    echo Error: PowerShell is required. Install PowerShell 5.1+ or use run.ps1 directly.
    exit /b 1
)

if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" -Command %1 %2 %3 %4 %5 %6 %7 %8 %9
)

endlocal & exit /b %ERRORLEVEL%
