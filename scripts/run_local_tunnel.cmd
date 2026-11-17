@echo off
rem Presentation backup: the whole swipe app on this laptop behind a public Cloudflare Tunnel URL.
rem Needs: bundle_swipe\ (training\16_export_swipe.py), web\dist\ (cd web ^&^& npm run build), and cloudflared.exe.
rem Prints a https://<random>.trycloudflare.com address; it works while this window stays open.
setlocal
cd /d "%~dp0.."
if not exist bundle_swipe\core.npz (echo bundle_swipe missing: run .venv\Scripts\python.exe training\16_export_swipe.py & exit /b 1)
if not exist web\dist\index.html (echo web\dist missing: cd web, then set VITE_API_URL=same-origin and run npm run build & exit /b 1)
set CLOUDFLARED=cloudflared
if exist D:\dev\tools\cloudflared.exe set CLOUDFLARED=D:\dev\tools\cloudflared.exe
start "AuraPath API" /min .venv\Scripts\python.exe -m uvicorn server.main:app --port 8000
timeout /t 5 /nobreak >nul
"%CLOUDFLARED%" tunnel --url http://localhost:8000
