# stop_all.ps1 — Kill all Stage 5 services
$pidsFile = Join-Path $PSScriptRoot "logs\.pids.json"
if (Test-Path $pidsFile) {
    $pids = Get-Content $pidsFile | ConvertFrom-Json
    foreach ($name in $pids.PSObject.Properties.Name) {
        $id = $pids.$name
        try {
            Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped $name (PID $id)"
        } catch {}
    }
    Remove-Item $pidsFile -ErrorAction SilentlyContinue
} else {
    # Fallback: kill by port
    @(10000, 10100, 10101, 10102, 10103) | ForEach-Object {
        $port = $_
        $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($conn) {
            $pid = $conn.OwningProcess | Select-Object -First 1
            if ($pid) {
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                Write-Host "Killed process on port $port (PID $pid)"
            }
        }
    }
}
Write-Host "Done."
