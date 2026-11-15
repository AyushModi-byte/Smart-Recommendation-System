@echo off
rem Copy experiment results (small JSON only, no weights) from the DGX to .\runs
rem Run from the repo root. Needs the "dgx" SSH alias (see docs\DGX_WORKFLOW.md).
setlocal
set REMOTE=dgx
set RDIR=btp/Smart-Recommendation-System

if not exist runs mkdir runs
ssh %REMOTE% "cd %RDIR% && tar czf - --ignore-failed-read runs/*/*.json" | tar xzf - -C .
dir /b runs
endlocal
