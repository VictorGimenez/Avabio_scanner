@echo off

call conda activate 3Dbce_env_win

echo Compilando...
python -m compileall -b .

@REM echo Removendo .py
@REM del /s *.py

echo Pronto

pause