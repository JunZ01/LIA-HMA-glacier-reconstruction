param(
    [int]$MaxWorkers = 12,
    [int]$PollSeconds = 15
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$pythonExe = "D:\Anaconda\envs\oggm_env2\python.exe"
$configRoot = Join-Path $projectRoot "Himalaya_greatGlacier\With_Rec_climate_organized\ice_surface_reconstruction\config\synthetic_lia_system_standard_inversion_v1"
$logRoot = Join-Path $projectRoot "w493continuous_batch34_queue_v1"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

$jobs = @(
    @{
        Name = "steady_1631_parent"
        Module = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_conservative_steady_screen_overlay_v1"
        Args = @("--config", (Join-Path $configRoot "steady_conservative_junction_1631_parent_batch34_v1.json"), "--delta-t", "-1.5", "--output-root", "w481steady1631parentv1")
    },
    @{
        Name = "steady_3175_parent_mindt30"
        Module = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_conservative_steady_screen_mindt30_v1"
        Args = @("--config", (Join-Path $configRoot "steady_conservative_junction_3175_parent_batch34_mindt30_v1.json"), "--delta-t", "1.5", "--output-root", "w482steady3175parentmindt30v1")
    },
    @{
        Name = "steady_3904_parent"
        Module = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_conservative_steady_screen_overlay_v1"
        Args = @("--config", (Join-Path $configRoot "steady_conservative_junction_3904_parent_batch34_v1.json"), "--delta-t", "1.5", "--output-root", "w483steady3904parentv1")
    },
    @{
        Name = "regularize_3137"
        Module = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.regularize_synthetic_lia_network_surface_v1"
        Args = @("--config", (Join-Path $configRoot "network_surface_regularization_3137_batch34_v1.json"))
    },
    @{
        Name = "history_3910_endp15"
        Module = "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.run_conservative_historical_ramp_overlay_v1"
        Args = @("--config", (Join-Path $configRoot "historical_conservative_3910_endp15_batch34_mindt30_v1.json"), "--end-delta-t", "1.5", "--output-root", "w490hist3910rampendp15mindt30v1")
    }
)

$manifest = Join-Path $logRoot "launch_manifest.csv"
"name,launched_at,pid,stdout,stderr" | Set-Content -LiteralPath $manifest -Encoding UTF8

foreach ($job in $jobs) {
    while (@(Get-Process -Name python -ErrorAction SilentlyContinue).Count -ge $MaxWorkers) {
        Start-Sleep -Seconds $PollSeconds
    }

    $stdout = Join-Path $logRoot ($job.Name + ".stdout.log")
    $stderr = Join-Path $logRoot ($job.Name + ".stderr.log")
    $arguments = @("-m", $job.Module) + $job.Args
    $process = Start-Process -FilePath $pythonExe -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $stamp = (Get-Date).ToString("o")
    ('"{0}","{1}",{2},"{3}","{4}"' -f $job.Name, $stamp, $process.Id, $stdout, $stderr) | Add-Content -LiteralPath $manifest -Encoding UTF8
    Start-Sleep -Seconds 2
}

$done = Join-Path $logRoot "queue_released.json"
$payload = @{
    status = "all-registered-jobs-released"
    released_count = $jobs.Count
    released_at = (Get-Date).ToString("o")
    max_workers = $MaxWorkers
} | ConvertTo-Json
Set-Content -LiteralPath $done -Value $payload -Encoding UTF8
