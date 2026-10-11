from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DIR_DATOS = RAIZ / "datos"
DIR_ACTIVACIONES = RAIZ / "activaciones"
DIR_SONDAS = RAIZ / "sondas_entrenadas"
DIR_FIGURAS = RAIZ / "figuras"
DIR_ESCENARIOS = RAIZ / "escenarios"

for _d in (DIR_DATOS, DIR_ACTIVACIONES, DIR_SONDAS, DIR_FIGURAS, DIR_ESCENARIOS):
    _d.mkdir(parents=True, exist_ok=True)

ARCHIVO_DATASET = DIR_DATOS / "dataset_sondas.csv"

MODELOS = {
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "capa": None,
        "ruta_capas": "model.layers",
        "necesita_token": False,
        "notas": "Apache 2.0, fuerte en español. Principal.",
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "capa": 16,
        "ruta_capas": "model.layers",
        "necesita_token": True,
        "notas": "Comparación directa con Burger et al. 2024 (su modelo principal).",
    },
    "smollm3-3b": {
        "hf_id": "HuggingFaceTB/SmolLM3-3B",
        "capa": None,
        "ruta_capas": "model.layers",
        "necesita_token": False,
        "notas": "Fully-open, ligero. Ideal para el filtro en la laptop (12GB) y como 3er punto de universalidad.",
    },
}

MODELO_DEFECTO = "qwen3-8b"

PREFIJO_ENUNCIADO = ""
USAR_PLANTILLA_CHAT = False
DTYPE = "float16"
CUANTIZAR_4BIT = False

UMBRAL_ENGANO = 0.5
MAX_REINTENTOS = 2
ALPHA_STEERING = 8.0
SEMILLA = 42
