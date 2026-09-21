@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "PORT=8000"
set "PID="
set "CHROME_PROFILE=%TEMP%\3dbce_chrome_profile"

REM ==========================================================
REM Localiza o Google Chrome
REM ==========================================================

set "CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"

if not exist "%CHROME%" set "CHROME=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"

if not exist "%CHROME%" set "CHROME=%LocalAppData%\Google\Chrome\Application\chrome.exe"

if not exist "%CHROME%" goto CHROME_NOT_FOUND


REM ==========================================================
REM Procura processo ouvindo na porta 8000
REM ==========================================================

for /f "tokens=5" %%A in ('netstat -aon ^| findstr ":%PORT%" ^| findstr "LISTENING"') do set "PID=%%A"


REM Se encontrou servidor, vai fechar
if defined PID goto STOP_APP


REM Caso contrario, inicia
goto START_APP



:START_APP

echo.
echo ==========================================
echo Iniciando aplicacao...
echo ==========================================
echo.

call conda activate 3Dbce_env_win

start "" /B python launcher.py

echo Aguardando servidor iniciar...

timeout /t 2 /nobreak >nul

echo Abrindo Google Chrome...

start "" "%CHROME%" --app="http://localhost:%PORT%/" --user-data-dir="%CHROME_PROFILE%"

echo.
echo Aplicacao iniciada.
echo.

goto END



:STOP_APP

echo.
echo ==========================================
echo Aplicacao encontrada
echo ==========================================
echo.

echo Porta: %PORT%
echo PID do servidor: %PID%
echo.

echo Encerrando servidor...

taskkill /F /PID %PID% >nul 2>&1

echo Fechando janela do Chrome...

powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'chrome.exe' -and $_.CommandLine -like '*3dbce_chrome_profile*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo.
echo Servidor e navegador encerrados.
echo.

goto END



:CHROME_NOT_FOUND

echo.
echo ERRO: Google Chrome nao encontrado.
echo.

goto END



:END

pause
endlocal