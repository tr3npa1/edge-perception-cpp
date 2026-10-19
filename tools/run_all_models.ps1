<#
.SYNOPSIS
Build/test, then freshly evaluate every active artifact and supported backend.
.DESCRIPTION
No export, calibration, training, Docker, Git operation or public-doc overwrite.
Default is the full test split and three timing repetitions. -Smoke is diagnostic
only. Resume with the original options, -Resume and the printed ResultsDirectory.
Keep the terminal open; zero GPU usage is NOT completion (CPU work is included).
#>
[CmdletBinding()]
param(
    [string]$ResultsDirectory = '',
    [switch]$Resume,
    [switch]$Smoke,
    [switch]$CheckPathsOnly,
    [string]$ModelsDirectory = 'models',
    [string]$Weights = 'runs/train/yolo26m_bdd100k/weights/best.pt',
    [string]$Data = 'data/processed/bdd100k_yolo/bdd100k.yaml',
    [string]$BuildDirectory = 'build/int8-deployment',
    [string]$BuildConfiguration = 'benchmarks/release_20261011/validation/nvidia-build.json',
    [string[]]$RuntimeDirectories = @(),
    [ValidateRange(0,64)][int]$Device = 0
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}
if ($Resume -and -not $ResultsDirectory) { throw '-Resume requires the original -ResultsDirectory.' }
if (-not $ResultsDirectory) { $ResultsDirectory = 'benchmarks/all_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') }
$results = Resolve-ProjectPath $ResultsDirectory
$python = Resolve-ProjectPath '.venv/Scripts/python.exe'
$ModelsDirectory = Resolve-ProjectPath $ModelsDirectory
$Weights = Resolve-ProjectPath $Weights
$Data = Resolve-ProjectPath $Data
$BuildDirectory = Resolve-ProjectPath $BuildDirectory
$BuildConfiguration = Resolve-ProjectPath $BuildConfiguration
$RuntimeDirectories = @($RuntimeDirectories | ForEach-Object { Resolve-ProjectPath $_ })
$executable = Join-Path $BuildDirectory 'bin/Release/edge_perception.exe'
$arguments = @((Join-Path $PSScriptRoot 'all_models.py'), '--output', (Join-Path $results 'pipeline'),
    '--models-dir', $ModelsDirectory, '--weights', $Weights, '--data', $Data,
    '--executable', $executable, '--device', "$Device")
if ($Smoke) { $arguments += '--smoke' }
if ($Resume) { $arguments += '--resume' }
if ($CheckPathsOnly) {
    [ordered]@{ project_root = $projectRoot; python = $python; results = $results;
        models = $ModelsDirectory; weights = $Weights; data = $Data; build = $BuildDirectory;
        executable = $executable; build_configuration = $BuildConfiguration;
        runtime_directories = $RuntimeDirectories } | ConvertTo-Json
    exit 0
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing project Python: $python" }
if ((Test-Path -LiteralPath $results) -and -not $Resume) {
    throw "Existing evidence preserved; use -Resume or a NEW ResultsDirectory: $results"
}
if ($Resume -and -not (Test-Path -LiteralPath (Join-Path $results 'pipeline/manifest.json'))) {
    throw 'No all-model manifest to resume.'
}
# Validate canonical INT8 hashes and the complete inventory before spending time
# on preparation. This creates no outputs and does not start model execution.
& $python @arguments --check-only
if ($LASTEXITCODE -ne 0) { throw 'Artifact/dataset preflight failed.' }
$null = New-Item -ItemType Directory -Path $results -Force
$prepare = Join-Path $results ('prep_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$buildArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'run_fp16_matrix.ps1'),
    '-PrepareOnly', '-ResultsDirectory', $prepare, '-BuildDirectory', $BuildDirectory,
    '-BuildConfiguration', $BuildConfiguration, '-Engine', (Join-Path $ModelsDirectory 'engine/yolo26m_bdd100k_fp16.engine'),
    '-Device', "$Device")
if ($RuntimeDirectories.Count) { $env:EDGE_RUNTIME_DIRS = $RuntimeDirectories -join ';' }
& powershell @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Preparation failed. Evidence: $prepare" }
# Windows PowerShell 5.1 emits a JSON array as one pipeline object. Wrapping
# that pipeline in @() nests the array and joins PATH as "System.Object[]".
$runtime = Get-Content -LiteralPath (Join-Path $prepare 'runtime-directories.json') -Raw | ConvertFrom-Json
$env:Path = ($runtime -join ';') + ';' + $env:Path
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_OFFLINE = 'true'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MPLBACKEND = 'Agg'
& $python -m pytest (Join-Path $projectRoot 'tests/test_all_models.py') (Join-Path $projectRoot 'tests/test_int8_workflow.py') -q "--basetemp=$(Join-Path $prepare 'workflow-tests')"
if ($LASTEXITCODE -ne 0) { throw 'Workflow regression tests failed.' }
Write-Host "Evidence: $results"
Write-Host 'Full mode includes CPU inference and full-test accuracy for every supported path; allow many hours.'
Write-Host 'Do not change code/models/runtime while running. Wait for ALL-MODEL SUITE FINISHED, not zero GPU use.'
Write-Host "Resume: use the same command/options plus -Resume -ResultsDirectory `"$results`""
$savedPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    $global:LASTEXITCODE = $null
    & $python @arguments 2>&1 | Tee-Object -FilePath (Join-Path $results ('launcher_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log'))
    $code = $global:LASTEXITCODE
} finally { $ErrorActionPreference = $savedPreference }
Write-Host "Report: $results/pipeline/README.md"
if ($null -eq $code) { throw 'Runner did not return an exit code; inspect its log.' }
if ($code -eq 2) { Write-Host 'Finished with recorded failures/unsupported cells/parity differences; inspect the report. This is not an all-pass claim.' }
if ($Smoke) { Write-Host 'DIAGNOSTIC ONLY: do not publish these small-sample numbers as full benchmark results.' }
exit $code
