@echo off
echo Copying YOLOv12 Project Report files...
echo.

set SOURCE=C:\Users\sumit\.gemini\antigravity\brain\3e91d1ef-38d2-45bf-a3a9-012213334f99
set DEST=c:\Users\sumit\Sumit-Personal\college-projects\mini\fifth\archive\v12 - Copy\report

if not exist "%DEST%" mkdir "%DEST%"

echo Copying report chapters...
copy "%SOURCE%\*.md" "%DEST%\" >nul

echo.
echo ✓ All report files copied to: %DEST%
echo.
echo Files included:
dir /b "%DEST%\*.md"
echo.
echo ✅ Done! Check the 'report' folder in your project directory.
pause
