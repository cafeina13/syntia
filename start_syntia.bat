@echo off
REM Double-click this file to start Syntia.
REM Keep this window open while the bot runs; closing it stops the bot.

REM Move to this script's folder so .env, System_Prompt.md, etc. are found.
cd /d "%~dp0"

REM Run the bot with the venv's Python (the one that has all the libraries).
.venv\Scripts\python.exe bot.py

REM If the bot stops or crashes, keep the window open so you can read why.
echo.
echo --- Syntia has stopped. Press any key to close this window. ---
pause >nul
