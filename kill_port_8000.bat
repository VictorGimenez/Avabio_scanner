
@echo off
setlocal enabledelayedexpansion
 
set PORTA=8000
 
echo Procurando processos usando a porta %PORTA%...
echo.
 
set ENCONTROU=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :%PORTA%') do (
    set ENCONTROU=1
    echo Matando PID %%p
    taskkill /PID %%p /F >nul 2>&1
)
 
if %ENCONTROU%==0 (
    echo Nenhum processo encontrado na porta %PORTA%. Nada pra fazer.
) else (
    echo.
    echo Pronto. Porta %PORTA% liberada.
)
 
pause