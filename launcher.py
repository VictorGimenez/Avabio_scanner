import subprocess
import sys

RESTART_CODE = 99

"""Script para reexecução da aplicação em caso de reinicialização para que seja possível sua execução na mesma porta."""
while True:
    print("[LAUNCHER] iniciando Main_web_version.py...")
    resultado = subprocess.run([sys.executable, "Main_web_version.py"])
    if resultado.returncode != RESTART_CODE:
        print(f"[LAUNCHER] app.py encerrou (code={resultado.returncode}). Finalizando.")
        break
    print("[LAUNCHER] reinício solicitado, subindo de novo...")