@echo off
rem Copy the data the v2 pipeline needs (about 12 GB) from the laptop to the DGX.
rem Run from the repo root. Needs the "dgx" SSH alias (see docs\DGX_WORKFLOW.md).
rem Nothing is deleted or changed locally.
setlocal
set REMOTE=dgx
set RDIR=Smart-Recommendation-System/brain

ssh %REMOTE% "mkdir -p %RDIR%" || exit /b 1
scp -r brain\ids %REMOTE%:%RDIR%/ || exit /b 1
scp brain\*.npy brain\*.csv brain\*.json brain\*.pth %REMOTE%:%RDIR%/ || exit /b 1
ssh %REMOTE% "du -sh %RDIR% && ls %RDIR% | wc -l"
endlocal
