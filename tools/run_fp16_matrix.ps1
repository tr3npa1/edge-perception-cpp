<#
.SYNOPSIS
Build, test and measure the two missing FP16 streaming configurations.
.DESCRIPTION
Default: native depth 3 and fused depth 1, plus separate sequential latency.
-FullMatrix measures all four streaming cells in a fresh matched run.
-Smoke uses 3 images, 3 warmups, 6 measured frames and one repetition; its
results are diagnostic only. No historical output is overwritten or reused.
SDK defaults can be recovered from this workstation's previous build record;
EDGE_* environment variables and -RuntimeDirectories override those hints.
#>
[CmdletBinding()]
param(
    [switch]$FullMatrix,
    [switch]$Smoke,
    [switch]$PrepareOnly,
    [string]$Engine = 'models/engine/yolo26m_bdd100k_fp16.engine',
    [string]$Source = 'data/raw/bdd100k/bdd100k/val/images',
    [string]$ResultsDirectory = '',
    [string]$BuildDirectory = 'build/fp16-matrix',
    [string]$BuildConfiguration = 'benchmarks/release_20261011/validation/nvidia-build.json',
    [string[]]$RuntimeDirectories = @(),
    [ValidateRange(0, 64)][int]$Device = 0,
    [ValidateRange(1, 32)][int]$BuildJobs = 4
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
function Absolute-Path([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}
if (-not $ResultsDirectory) {
    $ResultsDirectory = 'benchmarks/fp16_matrix_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
}
$results = Absolute-Path $ResultsDirectory
if (Test-Path -LiteralPath $results) { throw "Use a NEW ResultsDirectory; preserving existing evidence: $results" }
$null = New-Item -ItemType Directory -Path $results
$utf8 = New-Object System.Text.UTF8Encoding($false)
$steps = New-Object System.Collections.Generic.List[object]
$status = [ordered]@{ state = 'preparing'; diagnostic_only = [bool]$Smoke; steps = @(); error = $null }
function Save-Status {
    $status.updated_utc = [DateTime]::UtcNow.ToString('o')
    $status.steps = @($steps.ToArray())
    [IO.File]::WriteAllText((Join-Path $results 'preparation.json'), ($status | ConvertTo-Json -Depth 12), $utf8)
}
function Run-Step([string]$Name, [string]$Program, [string[]]$Arguments) {
    Write-Host "$Name..."
    $record = [ordered]@{ name = $Name; program = $Program; arguments = $Arguments; started_utc = [DateTime]::UtcNow.ToString('o') }
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Native commands update the global automatic variable. A local copy
        # would shadow that update inside this function and hide the exit code.
        $global:LASTEXITCODE = $null
        & $Program @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $results "$Name.log")
        $code = $global:LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    $record.exit_code = $code
    $record.finished_utc = [DateTime]::UtcNow.ToString('o')
    $steps.Add($record)
    Save-Status
    if ($null -eq $code -or $code -ne 0) { throw "$Name failed ($code). See $results/$Name.log" }
}

try {
    $python = Absolute-Path '.venv/Scripts/python.exe'
    $enginePath = Absolute-Path $Engine
    $sourcePath = Absolute-Path $Source
    foreach ($required in @($python, $enginePath)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing prerequisite file: $required" }
    }
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) { throw "Missing image directory: $sourcePath" }
    if ((Get-Item -LiteralPath $enginePath).Length -eq 0) { throw 'Engine file is empty.' }
    $fixture = Get-ChildItem -LiteralPath $sourcePath -Recurse -File |
        Where-Object { $_.Extension.ToLowerInvariant() -in @('.jpg','.jpeg','.png','.bmp','.ppm','.tif','.tiff','.webp') } |
        Sort-Object FullName | Select-Object -First 1
    if ($null -eq $fixture) { throw 'No supported image fixture found.' }

    # Recover only toolchain/SDK hints; never reuse historical binaries/results.
    $hints = @{}
    if (Test-Path -LiteralPath $BuildConfiguration -PathType Leaf) {
        $previous = @(Get-Content -LiteralPath $BuildConfiguration -Raw | ConvertFrom-Json)
        foreach ($argument in $previous[0].arguments) {
            if ($argument -match '^-D([^=]+)=(.*)$') { $hints[$matches[1]] = $matches[2] }
        }
    }
    $aliases = @{
        EDGE_CXX_COMPILER = 'CMAKE_CXX_COMPILER'; EDGE_OPENCV_DIR = 'OpenCV_DIR';
        EDGE_ONNXRUNTIME_ROOT = 'ONNXRUNTIME_ROOT'; EDGE_ONNXRUNTIME_RUNTIME_DIR = 'ONNXRUNTIME_RUNTIME_DIR';
        EDGE_CUDA_ROOT = 'CUDAToolkit_ROOT'; EDGE_TENSORRT_ROOT = 'TENSORRT_ROOT';
        EDGE_CUDA_ARCHITECTURES = 'CMAKE_CUDA_ARCHITECTURES'
    }
    foreach ($name in $aliases.Keys) {
        $value = [Environment]::GetEnvironmentVariable($name, 'Process')
        if ($value) { $hints[$aliases[$name]] = $value }
    }
    if ($hints.ContainsKey('CUDAToolkit_ROOT')) {
        $hints.CMAKE_CUDA_COMPILER = Join-Path $hints.CUDAToolkit_ROOT 'bin/nvcc.exe'
    }
    if ($hints.ContainsKey('CMAKE_CXX_COMPILER')) {
        $compiler = $hints.CMAKE_CXX_COMPILER
        if (-not (Test-Path -LiteralPath $compiler -PathType Leaf)) {
            throw "MSVC compiler missing: $compiler. Set EDGE_CXX_COMPILER to a CUDA-supported cl.exe."
        }
        $hints.CMAKE_CUDA_HOST_COMPILER = $compiler
        if ($compiler -match '^(.*)[/\\]VC[/\\]Tools[/\\]MSVC[/\\](\d+\.\d+)\.\d+[/\\]') {
            $vcvars = Join-Path $matches[1] 'VC/Auxiliary/Build/vcvars64.bat'
            $toolset = $matches[2]
            $environmentLines = & cmd.exe /d /s /c "`"$vcvars`" -vcvars_ver=$toolset >nul && set"
            if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the saved MSVC toolset.' }
            foreach ($line in $environmentLines) {
                if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
            }
        }
    }
    foreach ($name in @('OpenCV_DIR','ONNXRUNTIME_ROOT','CUDAToolkit_ROOT','TENSORRT_ROOT')) {
        if (-not $hints.ContainsKey($name) -or -not (Test-Path -LiteralPath $hints[$name] -PathType Container)) {
            throw "Missing SDK $name. Set the matching EDGE_* environment variable or provide -BuildConfiguration with valid SDK paths."
        }
    }
    $runtime = @($RuntimeDirectories)
    if ($env:EDGE_RUNTIME_DIRS) { $runtime += $env:EDGE_RUNTIME_DIRS -split ';' }
    $runtime += @($hints.ONNXRUNTIME_RUNTIME_DIR, (Join-Path $hints.TENSORRT_ROOT 'lib'),
        (Join-Path $hints.TENSORRT_ROOT 'bin'), (Join-Path $hints.CUDAToolkit_ROOT 'bin'),
        (Join-Path (Split-Path $hints.OpenCV_DIR -Parent) 'bin'))
    # Optional ORT CUDA dependencies on the existing workstation; other hosts
    # can supply runtime locations without editing this script.
    $runtime += @('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64', 'E:/cpp-libs/cudnn9-runtime')
    $runtime = @($runtime | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Container) } | Select-Object -Unique)
    $env:Path = ($runtime -join ';') + ';' + $env:Path
    $env:PYTHONUNBUFFERED = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $env:YOLO_AUTOINSTALL = 'false'
    $env:YOLO_OFFLINE = 'true'
    foreach ($program in @('cmake','ctest','ninja','nvidia-smi','git','cl')) {
        if (-not (Get-Command $program -ErrorAction SilentlyContinue)) { throw "Missing build/runtime command: $program. Use a configured x64 developer environment." }
    }
    $build = Absolute-Path $BuildDirectory
    $arguments = @('--preset','nvidia-release','-B',$build,'-DCMAKE_TRY_COMPILE_CONFIGURATION=Release',
        "-DPython3_EXECUTABLE=$python", "-DEDGE_GPU_TEST_ENGINE=$enginePath", "-DEDGE_GPU_TEST_IMAGE=$($fixture.FullName)")
    foreach ($key in @('CMAKE_CXX_COMPILER','OpenCV_DIR','ONNXRUNTIME_ROOT','ONNXRUNTIME_RUNTIME_DIR',
                       'CMAKE_CUDA_COMPILER','CMAKE_CUDA_HOST_COMPILER','CUDAToolkit_ROOT','TENSORRT_ROOT','CMAKE_CUDA_ARCHITECTURES')) {
        if ($hints.ContainsKey($key)) { $arguments += "-D$key=$($hints[$key])" }
    }
    $status.state = 'building'
    Save-Status
    Run-Step 'configure' 'cmake' $arguments
    Run-Step 'build' 'cmake' @('--build',$build,'--config','Release','--parallel',"$BuildJobs")
    Run-Step 'ctest' 'ctest' @('--test-dir',$build,'-C','Release','--output-on-failure')
    # Each run owns a new test directory, avoiding stale system-temp permissions.
    Run-Step 'runner-tests' $python @('-m','pytest','tests/test_fp16_matrix.py','-q',
        "--basetemp=$(Join-Path $results 'runner-test-temporary')")
    $executable = Join-Path $build 'bin/Release/edge_perception.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw "Build did not produce $executable" }
    if ($PrepareOnly) {
        $status.state = 'prepared'
        Save-Status
        [IO.File]::WriteAllText((Join-Path $results 'runtime-directories.json'), (ConvertTo-Json -InputObject @($runtime)), $utf8)
        Write-Host "PREPARED: $executable (no benchmark measurements run)"
        exit 0
    }
    $arguments = @('tools/benchmark_fp16_matrix.py','--executable',$executable,'--engine',$enginePath,
        '--source',$sourcePath,'--output',(Join-Path $results 'matrix'),'--device',"$Device")
    if ($FullMatrix) { $arguments += '--full-matrix' }
    if ($Smoke) { $arguments += @('--source-frames','3','--warmup','3','--latency-iterations','6','--throughput-iterations','6','--repetitions','1') }
    $status.state = 'measuring'
    Save-Status
    Write-Host 'Keep the terminal open. Completion is the FINISHED message, not zero GPU utilization.'
    Run-Step 'matrix' $python $arguments
    $status.state = 'completed'
    Save-Status
    Write-Host "FINISHED. Report: $results/matrix/README.md"
    if ($Smoke) { Write-Host 'SMOKE ONLY: do not publish these short measurements as benchmark results.' }
} catch {
    $status.state = 'failed'
    $status.error = $_.Exception.Message
    Save-Status
    Write-Host "STOPPED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Evidence preserved: $results"
    exit 1
}
