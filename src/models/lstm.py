from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Descargar modelos entrenados
api.dataset_download_files(
    "vicentelorenzomarn/crypto-models",
    path="./models",
    unzip=True
)