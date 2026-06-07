# Run the full Transformer-GAT MedRec pipeline (setup -> train -> eval -> test -> UI).
# Clinical decision support only — not for autonomous prescribing.
#
# Usage:
#   .\run.ps1              # full pipeline + start dashboard
#   .\run.ps1 all
#   .\run.ps1 setup
#   .\run.ps1 train
#   .\run.ps1 eval
#   .\run.ps1 test
#   .\run.ps1 explain
#   .\run.ps1 ui
#   .\run.ps1 stop
#   .\run.ps1 all -SkipTrain
#   .\run.ps1 all -ForceTrain
#   .\run.ps1 ui -Port 8081

param(
    [Parameter(Position = 0)]
    [ValidateSet("all", "setup", "train", "eval", "test", "explain", "ui", "stop", "help")]
    [string]$Command = "all",

    [switch]$SkipTrain,
    [switch]$ForceTrain,
    [int]$Port = 8080,
    [string]$UiHost = "127.0.0.1",
    [string]$Config = "config.yaml",
    [string]$Checkpoint = "checkpoints/best.pt",
    [string]$UseSynthetic = "true",
    [switch]$PortExplicit
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Write-Log([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-BootstrapPython {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { return @{ Exe = "py"; Args = @("-3") } }
    throw "Python not found. Install Python 3.10+ from https://www.python.org/"
}

function Get-VenvPython {
    $venvPy = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) { return $venvPy }
    return Get-BootstrapPython
}

function Invoke-Python([string[]]$Args) {
    $py = $script:Python
    if ($py -is [hashtable]) {
        & $py.Exe @($py.Args + $Args)
    } else {
        & $py @Args
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Test-PortInUse([int]$PortNum) {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $PortNum)
    try {
        $listener.Start()
        $listener.Stop()
        return $false
    } catch {
        return $true
    }
}

function Find-FreePort([int]$Start = 8080, [int]$Max = 8099) {
    for ($p = $Start; $p -le $Max; $p++) {
        if (-not (Test-PortInUse $p)) { return $p }
    }
    return $null
}

function Get-PidOnPort([int]$PortNum) {
    try {
        $conn = Get-NetTCPConnection -LocalPort $PortNum -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($conn) { return $conn.OwningProcess }
    } catch {
        # Get-NetTCPConnection may require admin on older Windows
    }
    $line = netstat -ano | Select-String ":\s*$PortNum\s+.*LISTENING" | Select-Object -First 1
    if ($line) {
        $parts = ($line -split '\s+') | Where-Object { $_ -ne '' }
        return [int]$parts[-1]
    }
    return $null
}

function Stop-UiServer {
    $processId = Get-PidOnPort $script:UiPort
    if (-not $processId) {
        Write-Log "No process listening on port $($script:UiPort)"
        return
    }
    Write-Log "Stopping UI server (PID $processId) on port $($script:UiPort)"
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    if (Test-PortInUse $script:UiPort) {
        throw "Could not free port $($script:UiPort)"
    }
    Write-Log "Port $($script:UiPort) is free"
}

function Resolve-UiPort {
    if (-not (Test-PortInUse $script:UiPort)) { return }
    if ($script:PortExplicit) {
        $processId = Get-PidOnPort $script:UiPort
        Write-Error @"
Port $($script:UiPort) is already in use.
  Stop the old server:  .\run.ps1 stop -Port $($script:UiPort)
  Or pick another port: .\run.ps1 ui -Port 8081
$(if ($processId) { "  Process on port: PID $processId" })
"@
    }
    $newPort = Find-FreePort $script:UiPort 8099
    if (-not $newPort) {
        throw "No free port between $($script:UiPort) and 8099. Run: .\run.ps1 stop"
    }
    Write-Log "Port $($script:UiPort) is in use — using $newPort instead (or run: .\run.ps1 stop)"
    $script:UiPort = $newPort
}

function Step-Setup {
    Write-Log "Creating virtual environment (if needed)"
    $venvPy = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) {
        $boot = Get-BootstrapPython
        if ($boot -is [hashtable]) {
            & $boot.Exe @($boot.Args + "-m" "venv" (Join-Path $Root ".venv"))
        } else {
            & $boot -m venv (Join-Path $Root ".venv")
        }
        if (-not (Test-Path $venvPy)) { throw "Failed to create .venv" }
    }
    $script:Python = $venvPy
    Write-Log "Installing dependencies"
    Invoke-Python -m pip install -q --upgrade pip
    Invoke-Python -m pip install -q -r (Join-Path $Root "requirements.txt")
    Write-Host "Setup complete. Python: $($script:Python)"
}

function Step-Train {
    $ckptPath = Join-Path $Root $Checkpoint
    if (-not $ForceTrain -and (Test-Path $ckptPath)) {
        if ($SkipTrain) {
            Write-Log "Skipping training (checkpoint exists: $Checkpoint). Use -ForceTrain to retrain."
            return
        }
    } elseif ($SkipTrain) {
        Write-Log "No checkpoint found — training anyway."
    }
    Write-Log "Training model (synthetic data)"
    Invoke-Python (Join-Path $Root "src\main.py") --config $Config --mode train --use_synthetic $UseSynthetic
}

function Step-Eval {
    $ckptPath = Join-Path $Root $Checkpoint
    if (-not (Test-Path $ckptPath)) {
        throw "Checkpoint not found at $Checkpoint. Run: .\run.ps1 train"
    }
    Write-Log "Evaluating on test set"
    Invoke-Python (Join-Path $Root "src\evaluate.py") --checkpoint $Checkpoint --use_synthetic $UseSynthetic
}

function Step-Test {
    Write-Log "Running unit tests"
    $pytest = Join-Path $Root ".venv\Scripts\pytest.exe"
    if (Test-Path $pytest) {
        & $pytest (Join-Path $Root "tests") -v
    } else {
        Invoke-Python -m pytest (Join-Path $Root "tests") -v
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Step-Explain {
    $ckptPath = Join-Path $Root $Checkpoint
    if (-not (Test-Path $ckptPath)) {
        throw "Checkpoint not found at $Checkpoint. Run: .\run.ps1 train"
    }
    Write-Log "Running SHAP explanation (patient 0)"
    Invoke-Python (Join-Path $Root "src\explain.py") --checkpoint $Checkpoint --use_synthetic $UseSynthetic --patient_idx 0
}

function Step-Ui {
    $ckptPath = Join-Path $Root $Checkpoint
    if (-not (Test-Path $ckptPath)) {
        Write-Warning "No checkpoint at $Checkpoint — UI will load but predictions may be untrained."
    }
    Resolve-UiPort
    Write-Log "Starting dashboard at http://${script:UiHost}:$($script:UiPort)"
    Write-Log "Press Ctrl+C to stop the server."
    $env:UI_HOST = $script:UiHost
    $env:UI_PORT = "$($script:UiPort)"
    $pyCode = @"
import os, uvicorn
from pathlib import Path
root = Path(r'$Root')
os.chdir(root)
import sys
sys.path.insert(0, str(root))
uvicorn.run('ui.server:app', host=os.environ.get('UI_HOST', '127.0.0.1'), port=int(os.environ.get('UI_PORT', '8080')), app_dir=str(root))
"@
    Invoke-Python -c $pyCode
}

function Step-All {
    Step-Setup
    Step-Train
    Step-Eval
    Step-Test
    Write-Host ""
    Write-Log "Pipeline finished. Launching web dashboard..."
    Write-Host "    Open http://${script:UiHost}:$($script:UiPort) in your browser."
    Write-Host "    Clinical decision support only — physician review required."
    Write-Host ""
    Step-Ui
}

function Show-Help {
    Get-Content $PSCommandPath -TotalCount 18 | Select-Object -Skip 1
}

# Parse -Port as explicit if user passed it (default binding doesn't set PortExplicit)
if ($PSBoundParameters.ContainsKey("Port")) {
    $script:PortExplicit = $true
} else {
    $script:PortExplicit = $false
}

$script:UiPort = $Port
$script:UiHost = $UiHost
$script:Python = Get-VenvPython

switch ($Command) {
    "help" { Show-Help }
    "setup" { Step-Setup }
    "train" { Step-Setup; Step-Train }
    "eval" { Step-Setup; Step-Eval }
    "test" { Step-Setup; Step-Test }
    "explain" { Step-Setup; Step-Explain }
    "stop" { Stop-UiServer }
    "ui" { Step-Setup; Step-Ui }
    "all" { Step-All }
    default {
        Write-Error "Unknown command: $Command. Commands: all, setup, train, eval, test, explain, ui, stop"
    }
}
