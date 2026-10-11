import os
import gc
import argparse
import numpy as np
import pandas as pd

import config


def obtener_token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None


def cargar_modelo(nombre_modelo, cuantizar_4bit=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if nombre_modelo not in config.MODELOS:
        raise ValueError(f"Modelo desconocido: {nombre_modelo}. Opciones: {list(config.MODELOS)}")
    info = config.MODELOS[nombre_modelo]
    hf_id = info["hf_id"]
    token = obtener_token() if info["necesita_token"] else None
    if info["necesita_token"] and not token:
        raise RuntimeError(
            f"{hf_id} requiere token de HF. Acepta la licencia en su página y "
            f"define HF_TOKEN (variable de entorno o Secret de Colab)."
        )

    if cuantizar_4bit is None:
        cuantizar_4bit = config.CUANTIZAR_4BIT

    dtype = getattr(torch, config.DTYPE)
    print(f"Cargando {hf_id}  (4bit={cuantizar_4bit}, dtype={config.DTYPE})...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs = dict(token=token, output_hidden_states=True)
    if cuantizar_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "auto"

    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()

    n_capas = model.config.num_hidden_layers
    capa = info["capa"] if info["capa"] is not None else n_capas // 2
    print(f"  num_hidden_layers={n_capas}  ->  capa objetivo = {capa}")
    return model, tokenizer, info, capa


def _construir_entrada(textos):
    if config.USAR_PLANTILLA_CHAT:
        return textos
    return [config.PREFIJO_ENUNCIADO + t for t in textos]


def extraer(textos, model, tokenizer, capa, todas_capas=False, batch=8):
    import torch
    entradas = _construir_entrada(textos)
    n = len(entradas)
    n_capas = model.config.num_hidden_layers

    salida_unica = []
    salida_por_capa = {L: [] for L in range(n_capas + 1)}

    for ini in range(0, n, batch):
        lote = entradas[ini:ini + batch]
        if config.USAR_PLANTILLA_CHAT:
            lote = [tokenizer.apply_chat_template(
                        [{"role": "user", "content": t}],
                        tokenize=False, add_generation_prompt=False)
                    for t in lote]
        enc = tokenizer(lote, return_tensors="pt", padding=True, truncation=True,
                        max_length=128).to(model.device)
        with torch.no_grad():
            out = model(**enc)
        idx_ult = enc["attention_mask"].sum(dim=1) - 1
        b_range = torch.arange(enc["input_ids"].shape[0], device=model.device)

        if todas_capas:
            for L, hs in enumerate(out.hidden_states):
                vec = hs[b_range, idx_ult, :].float().cpu().numpy()
                salida_por_capa[L].append(vec)
        else:
            hs = out.hidden_states[capa]
            vec = hs[b_range, idx_ult, :].float().cpu().numpy()
            salida_unica.append(vec)

        del enc, out
        if (ini // batch) % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  procesados {min(ini + batch, n)}/{n}")

    if todas_capas:
        return {L: np.concatenate(v, axis=0) for L, v in salida_por_capa.items() if v}
    return np.concatenate(salida_unica, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=config.MODELO_DEFECTO, choices=list(config.MODELOS))
    ap.add_argument("--capa", type=int, default=None)
    ap.add_argument("--todas-capas", action="store_true")
    ap.add_argument("--4bit", dest="cuatro_bits", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(config.ARCHIVO_DATASET)
    textos = df["texto"].tolist()

    model, tokenizer, info, capa = cargar_modelo(args.modelo, cuantizar_4bit=args.cuatro_bits)
    if args.capa is not None:
        capa = args.capa
        print(f"  capa sobreescrita por argumento -> {capa}")

    print(f"\nExtrayendo activaciones de {len(textos)} enunciados...")
    if args.todas_capas:
        acts = extraer(textos, model, tokenizer, capa, todas_capas=True)
        ruta = config.DIR_ACTIVACIONES / f"acts_TODAS_{args.modelo}.npz"
        np.savez_compressed(ruta, **{f"capa_{L}": A for L, A in acts.items()})
        print(f"Guardado (todas las capas): {ruta}")
    else:
        acts = extraer(textos, model, tokenizer, capa, todas_capas=False)
        ruta = config.DIR_ACTIVACIONES / f"acts_{args.modelo}_capa{capa}.npy"
        np.save(ruta, acts)
        print(f"Guardado: {ruta}  shape={acts.shape}")

    np.save(config.DIR_ACTIVACIONES / "etiquetas.npy", df["etiqueta"].to_numpy())
    np.save(config.DIR_ACTIVACIONES / "polaridades.npy", df["polaridad"].to_numpy())
    np.save(config.DIR_ACTIVACIONES / "topicos.npy", df["topico"].to_numpy())
    print("Etiquetas/polaridades/tópicos guardados.")

    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
