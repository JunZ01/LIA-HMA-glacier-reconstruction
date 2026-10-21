param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputFile
)

$ErrorActionPreference = 'Stop'
$routingPath = Join-Path $ProjectRoot 'validation_data\results\paleo_surface_validation_v0\lia_191_controller_v1\diagnostics\core191_routing_v1\core191_routing_table.csv'
$templatePath = Join-Path $PSScriptRoot 'core191_routing_scatter_visualization_template.html'
$routes = @(
    'neither',
    'rgi_only',
    'lia_only',
    'endpoint_tradeoff_melt_f_delta_t_closed'
)

$points = [System.Collections.Generic.List[object]]::new()
foreach ($row in (Import-Csv -LiteralPath $routingPath)) {
    if ($row.primary_route -notin $routes) {
        continue
    }
    $points.Add([pscustomobject]@{
        id = [int]$row.lia_unique_id
        route = [string]$row.primary_route
        eLia = [math]::Round([double]$row.min_e_lia_sep, 4)
        eRgi = [math]::Round([double]$row.min_e_rgi_sep, 4)
        deltaT = if ([string]::IsNullOrWhiteSpace($row.best_delta_t)) { $null } else { [math]::Round([double]$row.best_delta_t, 3) }
        amplitude = if ([string]::IsNullOrWhiteSpace($row.best_amplitude_ratio_to_target)) { $null } else { [math]::Round([double]$row.best_amplitude_ratio_to_target, 3) }
    })
}

if ($points.Count -ne 148) {
    throw "Expected 148 clean retreat systems, found $($points.Count)."
}

$json = $points | ConvertTo-Json -Depth 5 -Compress
$template = Get-Content -LiteralPath $templatePath -Raw -Encoding UTF8
$html = $template.Replace('__ROUTING_POINTS__', $json)
$outputDirectory = Split-Path -Parent $OutputFile
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}
Set-Content -LiteralPath $OutputFile -Value $html -Encoding UTF8
Write-Output "Wrote $OutputFile with $($points.Count) systems."
