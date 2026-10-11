import csv
import config

HECHOS = [
    ("Lima",            "es la capital de Perú",        "es la capital de Chile",        "geografia"),
    ("París",           "es la capital de Francia",     "es la capital de Italia",       "geografia"),
    ("Tokio",           "es la capital de Japón",       "es la capital de China",        "geografia"),
    ("Madrid",          "es la capital de España",      "es la capital de Portugal",     "geografia"),
    ("El río Amazonas", "está en Sudamérica",           "está en Europa",                "geografia"),
    ("El Sahara",       "es un desierto de África",     "es un desierto de Oceanía",     "geografia"),
    ("El Everest",      "es la montaña más alta",       "es el río más largo",           "geografia"),
    ("Machu Picchu",    "está en Perú",                 "está en Bolivia",               "geografia"),
    ("El volcán Misti", "está cerca de Arequipa",       "está cerca de Lima",            "geografia"),
    ("La Antártida",    "está cubierta de hielo",       "está cubierta de arena",        "geografia"),

    ("El agua",         "hierve a cien grados Celsius", "hierve a veinte grados Celsius","ciencia"),
    ("El Sol",          "es una estrella",              "es un planeta",                 "ciencia"),
    ("La Luna",         "orbita la Tierra",             "orbita Marte",                  "ciencia"),
    ("El hielo",        "es agua en estado sólido",     "es agua en estado gaseoso",     "ciencia"),
    ("La gravedad",     "atrae los objetos al suelo",   "empuja los objetos al cielo",   "ciencia"),
    ("La luz",          "viaja más rápido que el sonido","viaja más lento que el sonido","ciencia"),
    ("El oxígeno",      "es un gas a temperatura ambiente","es un metal sólido",         "ciencia"),
    ("El hidrógeno",    "es el elemento más abundante", "es el elemento más pesado",     "ciencia"),
    ("El diamante",     "está hecho de carbono",        "está hecho de queso",           "ciencia"),
    ("El cobre",        "conduce la electricidad",      "aísla la electricidad",         "ciencia"),

    ("El corazón",      "bombea sangre",                "bombea aire",                   "biologia"),
    ("Las plantas",     "realizan la fotosíntesis",     "se alimentan de carne",         "biologia"),
    ("Los pingüinos",   "son aves",                     "son reptiles",                  "biologia"),
    ("Las ballenas",    "son mamíferos",                "son peces",                     "biologia"),
    ("Las arañas",      "tienen ocho patas",            "tienen doce patas",             "biologia"),
    ("El murciélago",   "es un mamífero que vuela",     "es un tipo de pájaro",          "biologia"),
    ("La miel",         "la producen las abejas",       "la producen las hormigas",      "biologia"),
    ("Los gatos",       "son felinos",                  "son caninos",                   "biologia"),
    ("La piel",         "es el órgano más grande",      "es un órgano interno diminuto", "biologia"),
    ("El guepardo",     "es el animal terrestre más rápido","es el animal más lento",    "biologia"),

    ("Un triángulo",    "tiene tres lados",             "tiene cuatro lados",            "matematica"),
    ("Un cuadrado",     "tiene cuatro lados iguales",   "tiene tres lados redondos",     "matematica"),
    ("Un círculo",      "tiene trescientos sesenta grados","tiene ciento ochenta grados","matematica"),
    ("El número ocho",  "es un número par",             "es un número impar",            "matematica"),
    ("Cien",            "es mayor que cincuenta",       "es menor que cincuenta",        "matematica"),
    ("Un minuto",       "tiene sesenta segundos",       "tiene cien segundos",           "matematica"),
    ("Una docena",      "equivale a doce unidades",     "equivale a quince unidades",    "matematica"),
    ("Un siglo",        "tiene cien años",              "tiene diez años",               "matematica"),
    ("El sistema binario","usa los dígitos cero y uno", "usa los dígitos ocho y nueve",  "matematica"),
    ("Un año bisiesto", "tiene trescientos sesenta y seis días","tiene trescientos días","matematica"),

    ("La Gran Muralla", "está en China",                "está en Alemania",              "cultura"),
    ("La Torre Eiffel", "está en París",                "está en Londres",               "cultura"),
    ("Las pirámides de Guiza","están en Egipto",         "están en Canadá",               "cultura"),
    ("Einstein",        "fue un físico",                "fue un pintor",                 "cultura"),
    ("La penicilina",   "la descubrió Fleming",         "la descubrió Newton",           "cultura"),
    ("El piano",        "es un instrumento musical",    "es una fruta tropical",         "cultura"),
    ("El ajedrez",      "se juega en un tablero",       "se juega con dados",            "cultura"),
    ("Australia",       "es país y continente",         "es una isla de Oceanía pequeña","cultura"),
    ("El petróleo",     "es un combustible fósil",      "es un recurso renovable diario","cultura"),
    ("El alfabeto español","tiene la letra eñe",        "no tiene vocales",              "cultura"),
]


def negar_es(verbo_predicado: str) -> str:
    return "no " + verbo_predicado


def generar_filas():
    filas = []
    for sujeto, correcto, incorrecto, topico in HECHOS:
        filas.append((f"{sujeto} {correcto}.", 1, +1, topico))
        filas.append((f"{sujeto} {negar_es(correcto)}.", 0, -1, topico))
        filas.append((f"{sujeto} {incorrecto}.", 0, +1, topico))
        filas.append((f"{sujeto} {negar_es(incorrecto)}.", 1, -1, topico))
    return filas


def main():
    filas = generar_filas()
    with open(config.ARCHIVO_DATASET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["texto", "etiqueta", "polaridad", "topico"])
        w.writerows(filas)

    n = len(filas)
    n_v = sum(1 for _, e, *_ in filas if e == 1)
    n_afirm = sum(1 for *_, p, _ in [(f[0], f[1], f[2], f[3]) for f in filas] if p == +1)
    print(f"Dataset creado en: {config.ARCHIVO_DATASET}")
    print(f"  {n} enunciados | {n_v} verdaderos / {n - n_v} falsos | "
          f"{n_afirm} afirmativos / {n - n_afirm} negados")
    print(f"  {len(set(f[3] for f in filas))} tópicos: {sorted(set(f[3] for f in filas))}")
    print("\nPrimeras 8 filas (un hecho completo + medio):")
    for fila in filas[:8]:
        print("  ", fila)


if __name__ == "__main__":
    main()
