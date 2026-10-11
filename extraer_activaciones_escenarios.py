import argparse
import re
import numpy as np

import config


def limpiar_respuesta(t):
    t = str(t).strip()
    t = re.sub(r'^\**\s*(T[úu]|Respuesta)\s*:?\**\s*', '', t).strip()
    t = t.replace("**", "").strip().strip('"').strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=config.MODELO_DEFECTO, choices=list(config.MODELOS))
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--capa", type=int, default=None)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--4bit", dest="cuatro_bits", action="store_true")
    args = ap.parse_args()

    import torch, pandas as pd
    from extraer_activaciones import cargar_modelo
    model, tok, info, capa_auto = cargar_modelo(args.modelo, cuantizar_4bit=args.cuatro_bits)
    capa = args.capa if args.capa is not None else capa_auto
    print(f"Extrayendo activaciones en la capa {capa}")

    filas = []
    for ruta in args.csv:
        df = pd.read_csv(ruta, encoding="utf-8-sig")
        n_ok = 0
        for _, r in df.iterrows():
            etq = str(r.get("etiqueta_manual", "")).strip().lower()
            if etq not in ("mentira", "honesta"):
                continue
            resp = limpiar_respuesta(r["respuesta"])
            if not resp:
                continue
            prompt = str(r["prompt"]).replace(" | ", "\n")
            filas.append((str(r["id"]), 1 if etq == "mentira" else 0, prompt, resp))
            n_ok += 1
        print(f"  {ruta}: {n_ok} filas usables")

    if not filas:
        raise SystemExit("[!] No hay filas 'mentira'/'honesta' con respuesta. "
                         "¿Etiquetaste el CSV?")

    X, y, grupos = [], [], []
    for i, (eid, lab, prompt, resp) in enumerate(filas):
        try:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt", return_dict=True)
        except TypeError:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
        pre = enc["input_ids"]
        rid = tok(resp, add_special_tokens=False, return_tensors="pt")["input_ids"]
        full = torch.cat([pre, rid], dim=1).to(model.device)
        with torch.no_grad():
            o = model(full, output_hidden_states=True)
        act = o.hidden_states[capa][0, -1, :].float().cpu().numpy()
        X.append(act); y.append(lab); grupos.append(eid)
        if (i + 1) % 20 == 0:
            print(f"  procesadas {i+1}/{len(filas)}")

    X = np.array(X); y = np.array(y); grupos = np.array(grupos)
    salida = args.salida or str(config.DIR_ACTIVACIONES / f"engano_{args.modelo}_capa{capa}.npz")
    np.savez(salida, X=X, y=y, grupos=grupos, capa=capa)
    print(f"\nGuardado: {salida}")
    print(f"  {len(y)} ejemplos | mentiras={int(y.sum())} | honestas={int((y==0).sum())}"
          f" | escenarios={len(set(grupos.tolist()))}")
    if y.sum() < 10:
        print("  [i] Pocas mentiras: el resultado será exploratorio (alta varianza).")


if __name__ == "__main__":
    main()
