# Copy YOLOv12 Project Report to Project Directory
# This script copies all report files from artifacts to your project

$sourceDir = "C:\Users\sumit\.gemini\antigravity\brain\3e91d1ef-38d2-45bf-a3a9-012213334f99"
$destDir = "c:\Users\sumit\Sumit-Personal\college-projects\mini\fifth\archive\v12 - Copy\report"

# Create report directory
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

# Copy all report files
Write-Host "Copying YOLOv12 Project Report files..." -ForegroundColor Green

Get-ChildItem -Path $sourceDir -Filter "*.md" | ForEach-Object {
    Copy-Item $_.FullName -Destination $destDir -Force
    Write-Host "  ✓ Copied $($_.Name)" -ForegroundColor Cyan
}

Write-Host "`nAll report files copied to: $destDir" -ForegroundColor Green
Write-Host "`nFiles included:" -ForegroundColor Yellow
Get-ChildItem -Path $destDir -Filter "*.md" | ForEach-Object {
    Write-Host "  - $($_.Name)"
}

Write-Host "`n✅ Done! You can now review the report files." -ForegroundColor Green
