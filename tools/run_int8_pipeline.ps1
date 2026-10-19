<#
.SYNOPSIS
Prepare, quantize, build, validate, evaluate and benchmark INT8 with checkpoints.
.DESCRIPTION
Default is a full, user-run PTQ workflow. -Smoke reuses existing artifacts and
only checks four quality/parity images and six timing frames per cell.
-Resume -ResultsDirectory <existing> reuses hash-verified stages/chunks.
-Mode qat requires an independently trained QDQ ONNX and provenance; it does
not automatically train or pretend that PTQ is QAT.
-CheckPathsOnly resolves paths without creating directories or executing work.
#>
[CmdletBinding()]
param(
    [string]$ResultsDirectory = '',
    [switch]$Resume,
    [switch]$Smoke,
    [switch]$Reexport,
    [switch]$CheckPathsOnly,
    [ValidateSet('ptq','qat')][string]$Mode = 'ptq',
    [string]$QatOnnx = '',
    [string]$QatProvenance = '',
    [string]$ReuseInt8Engine = '',
    [string]$ReuseInt8Onnx = '',
    [string]$Weights = 'runs/train/yolo26m_bdd100k/weights/best.pt',
    [string]$Data = 'data/processed/bdd100k_yolo/bdd100k.yaml',
    [string]$Fp32Onnx = 'models/onnx/yolo26m_bdd100k_fp32.onnx',
    [string]$Fp16Engine = 'models/engine/yolo26m_bdd100k_fp16.engine',
    [string]$BuildDirectory = 'build/int8-deployment',
    [string]$BuildConfiguration = 'benchmarks/release_20261011/validation/nvidia-build.json',
    [string[]]$RuntimeDirectories = @(),
    [ValidateRange(1,70000)][int]$CalibrationImages = 1024,
    [ValidateRange(0,64)][int]$Device = 0,
    [ValidateRange(0.1,64)][double]$WorkspaceGiB = 2,
    [ValidateRange(0,100)][double]$MaxMapDrop = 2
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
function Resolve-ProjectPath([string]$Value) {
    # Set-Location does not update .NET's process working directory in Windows
    # PowerShell. Always anchor relative paths explicitly to this repository.
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}
if ($Resume -and -not $ResultsDirectory) { throw '-Resume requires the original -ResultsDirectory.' }
if (-not $ResultsDirectory) { $ResultsDirectory = 'benchmarks/int8_pipeline_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') }
$results = Resolve-ProjectPath $ResultsDirectory
$python = Resolve-ProjectPath '.venv/Scripts/python.exe'
$BuildDirectory = Resolve-ProjectPath $BuildDirectory
$BuildConfiguration = Resolve-ProjectPath $BuildConfiguration
$Weights = Resolve-ProjectPath $Weights
$Data = Resolve-ProjectPath $Data
$Fp32Onnx = Resolve-ProjectPath $Fp32Onnx
$Fp16Engine = Resolve-ProjectPath $Fp16Engine
if ($ReuseInt8Engine) { $ReuseInt8Engine = Resolve-ProjectPath $ReuseInt8Engine }
if ($ReuseInt8Onnx) { $ReuseInt8Onnx = Resolve-ProjectPath $ReuseInt8Onnx }
if ($QatOnnx) { $QatOnnx = Resolve-ProjectPath $QatOnnx }
if ($QatProvenance) { $QatProvenance = Resolve-ProjectPath $QatProvenance }
$RuntimeDirectories = @($RuntimeDirectories | ForEach-Object { Resolve-ProjectPath $_ })
if ($CheckPathsOnly) {
    [ordered]@{ project_root = $projectRoot; python = $python; results = $results;
        build = $BuildDirectory; build_configuration = $BuildConfiguration;
        weights = $Weights; data = $Data; fp32_onnx = $Fp32Onnx; fp16_engine = $Fp16Engine;
        reuse_engine = $ReuseInt8Engine; reuse_onnx = $ReuseInt8Onnx;
        qat_onnx = $QatOnnx; qat_provenance = $QatProvenance; runtime_directories = $RuntimeDirectories
    } | ConvertTo-Json
    exit 0
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing project Python: $python" }
if ((Test-Path -LiteralPath $results) -and -not $Resume) { throw "Existing evidence is preserved; use -Resume or a NEW ResultsDirectory: $results" }
if ($Resume -and -not (Test-Path -LiteralPath (Join-Path $results 'pipeline/manifest.json'))) { throw 'No pipeline manifest to resume.' }
if ($Mode -eq 'qat' -and (-not $QatOnnx -or -not $QatProvenance)) {
    throw 'QAT mode requires -QatOnnx and -QatProvenance from an actual supported QAT training/export workflow. No automatic training is implemented.'
}
if ($Mode -eq 'ptq' -and ($QatOnnx -or $QatProvenance)) { throw 'Use -Mode qat for a QAT artifact.' }
$null = New-Item -ItemType Directory -Path $results -Force
$prepare = Join-Path $results ('preparation_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$buildArgs = @('-NoProfile','-ExecutionPolicy','Bypass','-File', (Join-Path $PSScriptRoot 'run_fp16_matrix.ps1'),
    '-PrepareOnly','-ResultsDirectory',$prepare,'-BuildDirectory',$BuildDirectory,
    '-BuildConfiguration',$BuildConfiguration,'-Engine',$Fp16Engine,'-Device',"$Device")
# Runtime SDK overrides are supplied to both child build and parent worker env.
if ($RuntimeDirectories.Count) { $env:EDGE_RUNTIME_DIRS = $RuntimeDirectories -join ';' }
& powershell @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Preparation failed. Evidence: $prepare" }
# Do not nest ConvertFrom-Json's array in @(): Windows PowerShell 5.1 would
# stringify the nested array as System.Object[] instead of DLL directories.
$runtime = Get-Content -LiteralPath (Join-Path $prepare 'runtime-directories.json') -Raw | ConvertFrom-Json
$env:Path = ($runtime -join ';') + ';' + $env:Path
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_OFFLINE = 'true'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MPLBACKEND = 'Agg'
& $python -m pytest (Join-Path $projectRoot 'tests/test_int8_workflow.py') -q "--basetemp=$(Join-Path $prepare 'int8-test-temporary')"
if ($LASTEXITCODE -ne 0) { throw 'INT8 workflow unit tests failed.' }
$arguments = @((Join-Path $PSScriptRoot 'int8_pipeline.py'),'--output',(Join-Path $results 'pipeline'),
    '--executable',(Join-Path $BuildDirectory 'bin/Release/edge_perception.exe'),
    '--weights',$Weights,'--data',$Data,'--fp32-onnx',$Fp32Onnx,'--fp16-engine',$Fp16Engine,
    '--calibration-images',"$CalibrationImages",'--device',"$Device",'--workspace-gib',"$WorkspaceGiB",'--max-map-drop',"$MaxMapDrop")
if ($Resume) { $arguments += '--resume' }
if ($Smoke) { $arguments += '--smoke' }
if ($Reexport) { $arguments += '--reexport' }
if ($ReuseInt8Engine) { $arguments += @('--reuse-int8-engine',$ReuseInt8Engine) }
if ($ReuseInt8Onnx) { $arguments += @('--reuse-int8-onnx',$ReuseInt8Onnx) }
if ($Mode -eq 'qat') { $arguments += @('--qat-onnx',$QatOnnx,'--qat-provenance',$QatProvenance) }
Write-Host "Evidence: $results"
Write-Host 'Leave this terminal open. Resume using the same options plus -Resume and this -ResultsDirectory.'
& $python @arguments
$code = $LASTEXITCODE
Write-Host "Report: $results/pipeline/README.md"
if ($code -eq 2) { Write-Host 'Completed with accuracy/parity acceptance failures; inspect the report.' }
exit $code
