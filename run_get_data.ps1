# Daily Quant Data Collection Script
# Run at 6 PM on weekdays

# Change to working directory
Set-Location -Path "C:\Users\Administrator\Documents\quant"

# Upgrade and start aktools server in background
Write-Host "Upgrading aktools and akshare..."
pip install aktools --upgrade -i https://pypi.org/simple
pip install akshare --upgrade -i https://pypi.org/simple

Write-Host "Starting aktools server..."
$aktools_process = Start-Process -FilePath "python" -ArgumentList "-m", "aktools" -PassThru

# Wait for server to start
Start-Sleep -Seconds 20

try {
    # Set date in yyyymmdd format
    $date = Get-Date -Format "yyyyMMdd"

    # Run R script
    Write-Host "`nScript started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "Running get_data.r..."
    Rscript get_data.r

    # Compress data folder
    Write-Host "`nCompressing data folder..."
    $zipFile = "data_$date.zip"
    Compress-Archive -Path ".\data\*" -DestinationPath $zipFile -Force

    # Move to Google Drive
    Write-Host "Moving to Google Drive..."
    Move-Item -Path $zipFile -Destination "G:\My Drive\quant\" -Force

    Write-Host "Process completed successfully!"
}
catch {
    Write-Error "An error occurred: $_"
}
finally {
    # Terminate aktools server
    Write-Host "Terminating aktools server..."
    Stop-Process -Id $aktools_process.Id -Force

    Write-Host "`nPress any key to close this window..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}