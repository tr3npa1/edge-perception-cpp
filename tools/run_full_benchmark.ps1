<#
.SYNOPSIS
Run the complete local release measurement suite and retain every result.
.DESCRIPTION
Uses the freshly verified CPU/NVIDIA builds and resumes completed measurements.
Runtime failures stay in the reports. Nothing is committed or uploaded.
#>
[CmdletBinding()]
param(
    [string]$ResultsDirectory = 'benchmarks/release_20261011'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
$executable = Join-Path $projectRoot 'build/benchmark-nvidia/bin/Release/edge_perception.exe'
$results = [IO.Path]::GetFullPath((Join-Path $projectRoot $ResultsDirectory))
$validation = Join-Path $results 'validation'
foreach ($required in @($python, $executable, "$validation/nvidia-build.json", "$validation/cpu-build.json")) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required verified build/evidence is missing: $required"
    }
}
$recordedBinary = Get-Content -LiteralPath "$validation/nvidia-executable.json" -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash -ne $recordedBinary.Hash) {
    throw 'The verified NVIDIA executable has changed. Rebuild and validate before measuring.'
}

$runtimeDirs = @(
    'E:/cpp-libs/onnxruntime-win-x64-gpu_cuda13-1.27.0/lib',
    'E:/cpp-libs/TensorRT-10.9.0.34/lib',
    'E:/cpp-libs/TensorRT-10.9.0.34/bin',
    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin',
    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64',
    'E:/cpp-libs/cudnn9-runtime',
    'C:/opencv/build/x64/vc16/bin'
)
$env:Path = ($runtimeDirs -join ';') + ';' + $env:Path
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_OFFLINE = 'true'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MPLBACKEND = 'Agg'

$utf8 = New-Object System.Text.UTF8Encoding($false)
$steps = New-Object System.Collections.Generic.List[object]
$statusFile = Join-Path $results 'suite_status.json'
$lockFile = Join-Path $results 'suite.lock'
$suiteLock = $null
$allComplete = $false

function Save-Status([string]$State, [string]$Stage, [string]$Message = '') {
    $status = [ordered]@{
        state = $State
        stage = $Stage
        updated_utc = [DateTime]::UtcNow.ToString('o')
        process_id = $PID
        message = $Message
        results_directory = $results
        steps = @($steps.ToArray())
    }
    [IO.File]::WriteAllText($statusFile, ($status | ConvertTo-Json -Depth 8), $utf8)
}

function Invoke-MeasurementStep([string]$Name, [string[]]$Arguments, [int[]]$AcceptedCodes = @(0)) {
    Save-Status 'running' $Name
    Write-Host "`n========== $Name ==========" -ForegroundColor Cyan
    $start = [DateTime]::UtcNow
    $savedPreference = $ErrorActionPreference
    # Native programs legitimately write progress/warnings to stderr.
    $ErrorActionPreference = 'Continue'
    try {
        & $python @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $results "$Name.log")
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    $steps.Add([ordered]@{name=$Name; exit_code=$code; started_utc=$start.ToString('o'); finished_utc=[DateTime]::UtcNow.ToString('o')})
    Save-Status 'running' $Name "Process exited with code $code."
    if ($code -notin $AcceptedCodes) {
        throw "$Name stopped with exit code $code. Results and logs are preserved in $results"
    }
}

try {
    $suiteLock = [IO.File]::Open($lockFile, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    Write-Host "Results: $results"
    Write-Host 'Keep this terminal open. GPU usage may reach zero during CPU tests.'
    Write-Host 'Completion is indicated by SUITE FINISHED, not GPU utilization.'

    $performance = Join-Path $results 'performance'
    $performanceArgs = @('tools/benchmark_release.py','--executable',$executable,'--output',$performance,
        '--source','data/raw/bdd100k/bdd100k/val/images','--models-dir','models',
        '--source-frames','128','--repetitions','3','--warmup','50',
        '--latency-iterations','300','--throughput-iterations','1000','--timeout-seconds','3600')
    if (Test-Path -LiteralPath "$performance/manifest.json") { $performanceArgs += '--resume' }
    Invoke-MeasurementStep 'performance' $performanceArgs @(0,2)
    $performanceManifest = Get-Content -LiteralPath "$performance/manifest.json" -Raw | ConvertFrom-Json
    if ($performanceManifest.status -ne 'completed') {
        throw "Performance matrix is not complete: $($performanceManifest.status)"
    }

    $accuracy = Join-Path $results 'accuracy'
    $evaluationArgs = @('tools/evaluate_release.py','--executable',$executable,'--output',$accuracy)
    if (Test-Path -LiteralPath "$accuracy/manifest.json") { $evaluationArgs += '--resume' }
    Invoke-MeasurementStep 'accuracy' ($evaluationArgs + @('--phase','accuracy'))
    # The first phase creates the manifest even when starting a fresh suite.
    if ('--resume' -notin $evaluationArgs) { $evaluationArgs += '--resume' }
    Invoke-MeasurementStep 'cpp-parity' ($evaluationArgs + @('--phase','cpp'))

    # Docker has a separate startup failure on this workstation. Preserve its
    # current status without opening Desktop, changing settings, or blocking GPU work.
    $dockerProbe = @'
import datetime,json,pathlib,subprocess,sys
record={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'container_benchmarks_executed':False}
try:
    p=subprocess.run(['docker','info','--format','{{.ServerVersion}}'],capture_output=True,text=True,timeout=30)
    record.update(engine_available=p.returncode==0,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    record['status']='requires_separate_container_validation' if p.returncode==0 else 'blocked_docker_engine_unavailable'
except (OSError,subprocess.TimeoutExpired) as error:
    record.update(engine_available=False,status='blocked_docker_engine_unavailable',error=str(error))
pathlib.Path(sys.argv[1]).write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Docker:',record['status'])
'@
    Invoke-MeasurementStep 'docker-status' @('-c',$dockerProbe,(Join-Path $validation 'docker-status.json'))

    $publication = Join-Path $projectRoot 'docs/results'
    if (Test-Path -LiteralPath $publication) {
        $publication = Join-Path $results ('review-' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
    }
    Invoke-MeasurementStep 'prepare-report' @('tools/publish_results.py','--source',$results,'--output',$publication)
    Save-Status 'finished' 'complete' 'All local model/backend attempts finished. Inspect reports for measured failures and Docker status.'
    $allComplete = $true
    Write-Host "`nSUITE FINISHED" -ForegroundColor Green
    Write-Host "Raw evidence: $results"
    Write-Host "Review report: $publication/README.md"
    Write-Host 'Finished means every scheduled attempt was recorded; check the failure and parity tables.'
} catch {
    if ($null -ne $suiteLock) { Save-Status 'stopped' 'error' $_.Exception.Message }
    Write-Host "SUITE STOPPED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Read $statusFile and the stage log before resuming."
} finally {
    if ($null -ne $suiteLock) { $suiteLock.Dispose() }
}
if (-not $allComplete) { exit 1 }
