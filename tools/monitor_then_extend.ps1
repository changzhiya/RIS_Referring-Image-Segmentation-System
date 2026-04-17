# Wait for ris_mvp train.py to finish, then resume for 5 more epochs (total = last.pt epoch + 5).
$ErrorActionPreference = "Stop"
$py = "D:\miniconda3\envs\ris_env\python.exe"
$mvp = "d:\NVIDIA GPU Computing Toolkit\CUDA\test\ris_mvp"
$resume = Join-Path $mvp "result\checkpoints_doc31\last.pt"
$data = Join-Path $mvp "refcoco_ready"
$trainIdx = Join-Path $mvp "refcoco_ready\splits\train.json"
$valIdx = Join-Path $mvp "refcoco_ready\splits\val.json"
$marker = [regex]::Escape("CUDA\test\ris_mvp")

Write-Host "[monitor_then_extend] watching for train.py under ris_mvp ..." -ForegroundColor Cyan
while ($true) {
    $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'train\.py' -and $_.CommandLine -match $marker }
    if (-not $procs) { break }
    $ids = @($procs | ForEach-Object { $_.ProcessId })
    Write-Host ("[monitor] train.py PIDs: " + ($ids -join ", ") + "  " + (Get-Date -Format "HH:mm:ss"))
    Start-Sleep -Seconds 45
}

Write-Host "[monitor] no matching train.py; reading checkpoint epoch ..." -ForegroundColor Green
if (-not (Test-Path $resume)) {
    Write-Error "Missing checkpoint: $resume"
}
$epochLine = & $py -c "import torch; print(int(torch.load(r'$resume', map_location='cpu', weights_only=False)['epoch']))"
$e = [int]$epochLine.Trim()
$total = $e + 5
Write-Host "[monitor] last.pt completed epoch=$e  ->  --epochs $total  (epochs $($e+1)..$total)" -ForegroundColor Green

$env:PYTHONUNBUFFERED = "1"
Set-Location $mvp
& $py train.py `
    --data-root $data `
    --train-index $trainIdx `
    --val-index $valIdx `
    --doc-stage1 `
    --epochs $total `
    --resume $resume `
    --no-batch-progress

Write-Host "[monitor_then_extend] second stage finished exit=$LASTEXITCODE" -ForegroundColor Cyan
