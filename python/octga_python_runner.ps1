param([int]$Workers = 30)
# OCT-GA Python revision jobs: waits for running workers, then launches hidden workers,
# keeps Windows awake, disables EcoQoS on the workers, and shows progress until all jobs are done.
$RUN = 'D:\MBayrakOCTGA\Kod\Python\run'
$RES = 'D:\MBayrakOCTGA\Revision_Results_Python_FULL'
$PY  = 'C:\Users\MehmetPC\.claude-science\orgs\803c2fb6-6340-452a-9ad7-316b301358b0\workspaces\b4430365-e71d-4d2c-af0c-2138b8846a95\.venv\octga\Scripts\python.exe'
if (-not (Test-Path $PY)) { Write-Host "Python ortami bulunamadi: $PY" -ForegroundColor Red; exit 1 }
Add-Type -Namespace W -Name K -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);
[DllImport("kernel32.dll")] public static extern IntPtr OpenProcess(uint a, bool i, int pid);
[DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
[DllImport("kernel32.dll")] public static extern bool SetProcessInformation(IntPtr h, int cls, uint[] info, uint size);
"@
function Get-Workers { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_full.py*worker*' }) }
function Set-HighQoS {
  foreach ($p in Get-Workers) {
    $h = [W.K]::OpenProcess(0x1200, $false, [int]$p.ProcessId)
    if ($h -ne [IntPtr]::Zero) { [void][W.K]::SetProcessInformation($h, 4, [uint32[]](1,1,0), 12); [void][W.K]::CloseHandle($h) }
  }
}
function Awake { [void][W.K]::SetThreadExecutionState([uint32]2147483649) }
function Count { (Get-ChildItem "$RES\summaries" -Filter *.csv -ErrorAction SilentlyContinue).Count }

while ((Get-Workers).Count -gt 0) {
  Awake; Set-HighQoS
  Write-Host ("[{0}] Onceki isciler hala calisiyor ({1}); bitmeleri bekleniyor. Tamamlanan: {2}" -f (Get-Date -f 'HH:mm'), (Get-Workers).Count, (Count))
  Start-Sleep -Seconds 120
}
$env:PYTHONPATH = $RUN; $env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'; $env:NUMBA_NUM_THREADS = '1'; $env:OCTGA_RUNTAG = 'shared'
Set-Location $RUN
& $PY run_full.py count
for ($i = 1; $i -le $Workers; $i++) {
  Start-Process -FilePath $PY -ArgumentList 'run_full.py', 'worker', (100 + $i) -WorkingDirectory $RUN -WindowStyle Hidden `
    -RedirectStandardOutput "$RES\logs\bat_out$i.txt" -RedirectStandardError "$RES\logs\bat_err$i.txt"
  Start-Sleep -Milliseconds 300
}
Start-Sleep -Seconds 20
while ((Get-Workers).Count -gt 0) {
  Awake; Set-HighQoS
  Write-Host ("[{0}] Calisan isci: {1}  |  Tamamlanan is dosyasi: {2}" -f (Get-Date -f 'HH:mm'), (Get-Workers).Count, (Count))
  Start-Sleep -Seconds 60
}
[void][W.K]::SetThreadExecutionState([uint32]2147483648)
& $PY run_full.py count
Write-Host 'TUM ISLER BITTI.' -ForegroundColor Green
