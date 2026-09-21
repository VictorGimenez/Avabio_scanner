@echo off

echo Instalando Python, PostgreSQL e Miniconda
:: Verifica se o Winget existe
where winget >nul 2>&1
if errorlevel 1 (
    echo Winget nao encontrado.
    echo Atualize o App Installer pela Microsoft Store.
    pause
    exit /b 1
)

:: ----------------------------
:: Python
:: ----------------------------
echo Verificando Python...

where python >nul 2>&1
if errorlevel 1 (
    echo Instalando Python...
    winget install --id Python.Python.3.12 -e ^
        --accept-package-agreements ^
        --accept-source-agreements
) else (
    echo Python ja instalado.
)

echo.

:: ----------------------------
:: Miniconda
:: ----------------------------
echo Verificando Miniconda...

where conda >nul 2>&1
if errorlevel 1 (
    echo Instalando Miniconda...
    winget install --id Anaconda.Miniconda3 -e ^
        --accept-package-agreements ^
        --accept-source-agreements
) else (
    echo Miniconda ja instalado.
)

echo.

:: ----------------------------
:: PostgreSQL
:: ----------------------------
echo Verificando PostgreSQL...

where psql >nul 2>&1

if errorlevel 1 (
    echo Instalando PostgreSQL...

    winget install --id PostgreSQL.PostgreSQL.18 -e ^
        --accept-package-agreements ^
        --accept-source-agreements

    echo.
    echo Verificando instalacao...

    where psql >nul 2>&1

    if errorlevel 1 (
        echo ERRO: PostgreSQL nao foi instalado corretamente.
        pause
        exit /b 1
    ) else (
        echo PostgreSQL instalado com sucesso.
    )

) else (
    echo PostgreSQL ja instalado.
)

:: ----------------------------
:: CH341 Driver
:: ----------------------------
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Failure: Please run this script as Administrator.
    pause
    exit /b 1
)

echo Installing CH341/CH340 driver...

:: Check for setup.exe or CH341SER_Windows in current directory
if exist "%~dp0setup.exe" (
    "%~dp0setup.exe" /S
) else if exist "%~dp0CH341SER_Windows.EXE" (
    "%~dp0CH341SER_Windows.EXE" /S
) else (
    echo Error: setup.exe or CH341SER_Windows.exe not found in script directory.
    pause
    exit /b 1
)


echo Criando ambiente...

call conda activate base

conda env create -f 3Dbce_env_win.yml

pause