@echo off
cd /d "%~dp0"
set HJ_DISABLE_MODEL=1
set HJ_LAPTOP_BUILD=1
powershell -ExecutionPolicy Bypass -File "%~dp0START_HALAL_JORDAN.ps1"
