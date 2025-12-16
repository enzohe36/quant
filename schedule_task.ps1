# Setup Task Scheduler for Daily Quant Data Collection
# Run this script as Administrator

$TaskName = "Quant Data Collection"
$ScriptPath = "C:\Users\Administrator\Documents\quant\run_get_data.ps1"

# Create the action (with -NoExit to keep window open)
$Action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-ExecutionPolicy Bypass -NoExit -File `"$ScriptPath`""

# Create the trigger (6 PM on weekdays)
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 6:00PM

# Create the principal (run as current user with highest privileges)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

# Create the settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Register the scheduled task
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force

Write-Host "Scheduled task '$TaskName' has been created successfully!"
Write-Host "The task will run at 6:00 PM on weekdays (Monday-Friday)."