param([string]$Flag)
# Keeps Windows awake while the OCT-GA MATLAB run is active (no power settings are changed).
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS_SYSTEM = [uint32]2147483649   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$ES_CONTINUOUS        = [uint32]2147483648
$start = Get-Date; $absent = 0
while (Test-Path -LiteralPath $Flag) {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS_SYSTEM) | Out-Null
    $ml = Get-Process -Name MATLAB -ErrorAction SilentlyContinue
    if ($ml) { $absent = 0 } elseif (((Get-Date) - $start).TotalMinutes -gt 10) { $absent++ }
    if ($absent -ge 5) { break }          # MATLAB gone for ~5 min (window closed) -> stop
    Start-Sleep -Seconds 60
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
