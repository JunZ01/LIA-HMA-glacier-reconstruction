param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile
)

$ErrorActionPreference = 'Stop'

$templatePath = Join-Path $PSScriptRoot 'tradeoff_curve_visualization_template.html'
$resultRoot = Join-Path $ProjectRoot 'validation_data\results\paleo_surface_validation_v0'
$experiments = @(
    [pscustomobject]@{
        name = 'pilot'
        path = Join-Path $resultRoot 'lia_191_tradeoff_pilot_v1'
    },
    [pscustomobject]@{
        name = 'refine'
        path = Join-Path $resultRoot 'lia_191_tradeoff_refine_2259_v1'
    },
    [pscustomobject]@{
        name = 'warm'
        path = Join-Path $resultRoot 'lia_191_tradeoff_refine_2259_warm_v1'
    }
)

$baselineMeltF = @{
    '2259' = 13.056801434917734
    '4959' = 2.440466470988435
}

$panelOrder = @{
    '2259|pilot|0.80' = 10
    '2259|pilot|1.00' = 20
    '2259|pilot|1.20' = 30
    '2259|refine|0.85' = 40
    '2259|refine|0.90' = 50
    '2259|refine|0.95' = 60
    '2259|warm|0.80' = 70
    '4959|pilot|0.80' = 10
    '4959|pilot|1.00' = 20
    '4959|pilot|1.20' = 30
}

$seriesRecords = [System.Collections.Generic.List[object]]::new()
$targets = @{}

foreach ($experiment in $experiments) {
    $scoresPath = Join-Path $experiment.path 'scores.csv'
    if (-not (Test-Path -LiteralPath $scoresPath)) {
        continue
    }

    foreach ($score in (Import-Csv -LiteralPath $scoresPath)) {
        $system = [string]$score.lia_unique_id
        if (-not $baselineMeltF.ContainsKey($system)) {
            continue
        }

        $timeseriesPath = Join-Path $score.output_dir 'timeseries.csv'
        if (-not (Test-Path -LiteralPath $timeseriesPath)) {
            continue
        }

        if (-not $targets.ContainsKey($system)) {
            $targets[$system] = [pscustomobject]@{
                lia = [math]::Round(([double]$score.lia_target_m / 1000.0), 4)
                rgi = [math]::Round(([double]$score.rgi_target_m / 1000.0), 4)
                separation = [math]::Round(([double]$score.lia_rgi_separation_m / 1000.0), 4)
            }
        }

        $ratio = [double]$score.melt_f / [double]$baselineMeltF[$system]
        $ratioText = $ratio.ToString('0.00', [System.Globalization.CultureInfo]::InvariantCulture)
        $panelKey = "$system|$($experiment.name)|$ratioText"
        if (-not $panelOrder.ContainsKey($panelKey)) {
            continue
        }

        $points = [System.Collections.Generic.List[object]]::new()
        $forwardRows = Import-Csv -LiteralPath $timeseriesPath | Where-Object { $_.phase -eq 'forward' }
        foreach ($row in $forwardRows) {
            $year = [int][math]::Round([double]$row.year)
            if ((($year - 1470) % 2 -ne 0) -and ($year -ne 2000)) {
                continue
            }
            $points.Add([pscustomobject]@{
                y = $year
                l = [math]::Round(([double]$row.length_m / 1000.0), 3)
            })
        }

        $stable = ($score.spinup_converged -eq 'True') -and ($score.summary_status -eq 'ok')
        $statusLabel = if ($stable) { '稳定' } elseif ($score.spinup_converged -ne 'True') { '未收敛' } else { '越界' }
        $panelLabel = if ($experiment.name -eq 'warm') {
            '暖侧延伸 · melt_f ×0.80'
        } elseif ($experiment.name -eq 'refine') {
            "细化块 · melt_f ×$ratioText"
        } else {
            "初始块 · melt_f ×$ratioText"
        }

        $seriesRecords.Add([pscustomobject]@{
            system = $system
            panelKey = $panelKey
            panelLabel = $panelLabel
            panelOrder = [int]$panelOrder[$panelKey]
            deltaT = [math]::Round([double]$score.delta_t, 3)
            stable = $stable
            status = $statusLabel
            points = $points
        })
    }
}

$panels = [System.Collections.Generic.List[object]]::new()
foreach ($group in ($seriesRecords | Group-Object panelKey)) {
    $first = $group.Group[0]
    $panels.Add([pscustomobject]@{
        system = $first.system
        key = $first.panelKey
        label = $first.panelLabel
        order = $first.panelOrder
        series = @($group.Group | Sort-Object deltaT)
    })
}

$payload = [pscustomobject]@{
    targets = $targets
    panels = @($panels | Sort-Object system, order)
}

$json = $payload | ConvertTo-Json -Depth 12 -Compress
$template = Get-Content -LiteralPath $templatePath -Raw
$html = $template.Replace('__TRADEOFF_DATA__', $json)

$outputDirectory = Split-Path -Parent $OutputFile
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}
Set-Content -LiteralPath $OutputFile -Value $html -Encoding UTF8

$size = (Get-Item -LiteralPath $OutputFile).Length
Write-Output "Wrote $OutputFile ($size bytes; $($panels.Count) panels; $($seriesRecords.Count) curves)."

