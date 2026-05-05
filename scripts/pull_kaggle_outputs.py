import os
import sys
import shutil
import zipfile  # ← ¡necesitas importar esto!
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# UTF-8
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

# ---------------- CONFIG ----------------
KERNEL = "vicentelorenzomarn/crypto-lstm"
DOWNLOAD_DIR = Path("tmp/kaggle_output")
FINAL_DIR = Path("models")

api = KaggleApi()

FILES_TO_GET = [
    "best_BTC.pt",
    "best_ETH.pt",
    "best_SOL.pt",
    "best_XRP.pt",
    "best_AVAX.pt",
    "best_BTC_fold5.pt",
    "best_ETH_fold5.pt",
    "best_SOL_fold5.pt",
    "best_XRP_fold5.pt",
    "best_AVAX_fold5.pt"
]
DIRS_TO_GET = [
    "plots"
]
# ----------------------------------------

def authenticate_kaggle():
    """Autentica con kaggle.json (debe estar en ~/.kaggle/ o variable de entorno)"""
    try:
        api.authenticate()
        print("Autenticación Kaggle correcta.")
    except Exception as e:
        print(f"Error de autenticación: {e}")
        raise

def download_kaggle_output():
    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR)
    DOWNLOAD_DIR.mkdir(parents=True)

    print(f"Descargando output del kernel '{KERNEL}'...")
    # Este método descarga un zip que contiene toda la salida (working dir).
    # Lo guarda en DOWNLOAD_DIR
    api.kernels_output(kernel=KERNEL, path=str(DOWNLOAD_DIR))

    # Descomprimir todos los .zip encontrados
    print("Descomprimiendo archivos zip...")
    for zip_file in DOWNLOAD_DIR.rglob("*.zip"):
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(zip_file.parent)  # extrae junto al zip

def find_and_copy():
    if not FINAL_DIR.exists():
        FINAL_DIR.mkdir(parents=True)

    # Buscamos recursivamente en todos los archivos descomprimidos
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        root_path = Path(root)

        # Copiar archivos
        for file in files:
            if file in FILES_TO_GET:
                src = root_path / file
                dst = FINAL_DIR / file
                shutil.copy2(src, dst)
                print(f"Archivo copiado: {file}")

        # Copiar carpetas (por ejemplo 'plots')
        for d in dirs:
            if d in DIRS_TO_GET:
                src_dir = root_path / d
                dst_dir = FINAL_DIR / d
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                print(f"Carpeta copiada: {d}")

def clean_tmp():
    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR)
        print("Carpeta temporal eliminada.")

if __name__ == "__main__":
    authenticate_kaggle()
    download_kaggle_output()
    find_and_copy()
    clean_tmp()
    print("✅ Proceso completado.")