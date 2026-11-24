@echo off
setlocal
cd /d "%~dp0"
echo Recreate Figures 2, 3, 5, 6 and 7 using the saved CRSS analysis CSVs.
echo The script will fail if the required inputs do not exist.
python "6_plot_figures\plot_paper_figures.py"
if errorlevel 1 (
    echo Figure generation failed. Check the message above and README.md.
    exit /b 1
)
echo Figure generation completed. See outputs\6_plot_figures\.
endlocal
