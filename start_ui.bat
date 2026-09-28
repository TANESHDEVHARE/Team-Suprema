@echo off
rem Starts the local solver interface and opens it in the browser.
cd /d "%~dp0"
start "" http://127.0.0.1:8765
python tools\ui_server.py %*
