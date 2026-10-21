$ErrorActionPreference = 'Stop'
$package = 'E:\3spyderProject\1_LIA_himalaya_oggm\polim_cpu\tools\nsight-systems-2026.5.1\NsightSystems-2026.5.1.161-3889610.msi'
$outPath = 'E:\3spyderProject\1_LIA_himalaya_oggm\polim_cpu\tools\nsight-systems-2026.5.1\installer-inspection.json'
if (Test-Path -LiteralPath $outPath) { throw 'Refuse to overwrite installer inspection' }
$signature = Get-AuthenticodeSignature -LiteralPath $package
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike '*NVIDIA Corporation*') { throw 'Installer signature not verified' }
$installer = New-Object -ComObject WindowsInstaller.Installer
$database = $installer.GetType().InvokeMember('OpenDatabase', 'InvokeMethod', $null, $installer, @($package, 0))
function Read-MsiQuery([string]$query) {
    $view = $database.GetType().InvokeMember('OpenView', 'InvokeMethod', $null, $database, @($query))
    $view.GetType().InvokeMember('Execute', 'InvokeMethod', $null, $view, $null) | Out-Null
    $rows = [System.Collections.Generic.List[object]]::new()
    while ($null -ne ($rec = $view.GetType().InvokeMember('Fetch', 'InvokeMethod', $null, $view, $null))) {
        $n = $rec.GetType().InvokeMember('FieldCount', 'GetProperty', $null, $rec, $null)
        $row = @()
        for ($i = 1; $i -le $n; $i++) { $row += $rec.GetType().InvokeMember('StringData', 'GetProperty', $null, $rec, @($i)) }
        $rows.Add($row)
    }
    $view.GetType().InvokeMember('Close', 'InvokeMethod', $null, $view, $null) | Out-Null
    return ,$rows.ToArray()
}
$tables = Read-MsiQuery 'SELECT `Name` FROM `_Tables`'
$queries = [ordered]@{
    Properties = 'SELECT `Property`,`Value` FROM `Property`'
    Features = 'SELECT `Feature`,`Title`,`Description`,`Level` FROM `Feature`'
    Upgrade = 'SELECT * FROM `Upgrade`'
    ServiceInstall = 'SELECT * FROM `ServiceInstall`'
    CustomAction = 'SELECT * FROM `CustomAction`'
    InstallExecuteSequence = 'SELECT * FROM `InstallExecuteSequence`'
    DriverFiles = 'SELECT `File`,`FileName` FROM `File`'
}
$result = [ordered]@{ path = $package; bytes = (Get-Item -LiteralPath $package).Length; sha256 = (Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash; signature = [ordered]@{status = [string]$signature.Status; signer = $signature.SignerCertificate.Subject}; tables = $tables }
foreach ($key in $queries.Keys) {
    try {
        $r = Read-MsiQuery $queries[$key]
        if ($key -eq 'DriverFiles') { $r = @($r | Where-Object { $_[1] -match '\.(sys|inf|cat)$' }) }
        $result[$key] = $r
    } catch { $result[$key] = @{ unavailable = $_.Exception.Message } }
}
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $outPath -Encoding UTF8
Write-Output $outPath
