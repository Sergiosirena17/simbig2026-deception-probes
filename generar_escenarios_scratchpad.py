import argparse
import csv
import time
import numpy as np

import config
from generar_escenarios import ESCENARIOS


def _guardar_csv(filas, ruta):
    campos = list(filas[0].keys())
    for destino in (ruta, ruta.with_name(
            f"{ruta.stem}_{time.strftime('%Y%m%d_%H%M%S')}{ruta.suffix}")):
        try:
            with open(destino, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=campos)
                w.writeheader(); w.writerows(filas)
            return destino
        except PermissionError:
            print(f"  [!] No pude escribir en {destino.name} (¿abierto en Excel?). "
                  f"Pruebo con otro nombre...")
    raise PermissionError("No se pudo guardar en ninguna ruta.")


def _localizar_cierre_think(gen_ids, tokenizer):
    gen = gen_ids.tolist()
    tid = tokenizer.convert_tokens_to_ids("</think>")
    unk = tokenizer.unk_token_id
    if tid is not None and tid != unk and tid in gen:
        return gen.index(tid)
    txt = tokenizer.decode(gen_ids, skip_special_tokens=False)
    if "</think>" not in txt:
        return None
    pref = txt.split("</think>")[0] + "</think>"
    n = len(tokenizer.encode(pref, add_special_tokens=False))
    return min(max(0, n - 1), len(gen) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=config.MODELO_DEFECTO, choices=list(config.MODELOS))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--sonda", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--4bit", dest="cuatro_bits", action="store_true")
    args = ap.parse_args()

    import torch, joblib
    from extraer_activaciones import cargar_modelo
    model, tokenizer, info, capa = cargar_modelo(args.modelo, cuantizar_4bit=args.cuatro_bits)

    sonda_pkg = joblib.load(args.sonda)
    sonda = sonda_pkg["sonda"]
    capa = sonda_pkg["capa"]
    print(f"Sonda cargada (capa {capa}). Se puntuará razonamiento Y respuesta.")

    def generar_y_puntuar(prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            enc = tokenizer.apply_chat_template(
                msgs, return_tensors="pt", add_generation_prompt=True,
                return_dict=True, enable_thinking=True)
        except TypeError:
            enc = tokenizer.apply_chat_template(
                msgs, return_tensors="pt", add_generation_prompt=True,
                return_dict=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=True, temperature=0.6, top_p=0.95,
                                 pad_token_id=tokenizer.pad_token_id)

        gen_ids = out[0, n_in:]
        idx_think = _localizar_cierre_think(gen_ids, tokenizer)

        with torch.no_grad():
            o = model(out, output_hidden_states=True)
        H = o.hidden_states[capa][0]

        if idx_think is not None and idx_think > 0:
            pos_razon = n_in + idx_think - 1
            act_razon = H[pos_razon].float().cpu().numpy()
            cerro = True
        else:
            act_razon = H[-1].float().cpu().numpy()
            cerro = False
        act_resp = H[-1].float().cpu().numpy()

        p_razon = float(sonda.prob_engano(act_razon)[0])
        p_resp = float(sonda.prob_engano(act_resp)[0])

        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "</think>" in raw:
            razonamiento = raw.split("</think>")[0].replace("<think>", "").strip()
            respuesta = raw.split("</think>", 1)[1].strip()
        else:
            razonamiento = raw.replace("<think>", "").strip()
            respuesta = ""

        for sep in (".", "!", "?"):
            if sep in respuesta:
                respuesta = respuesta[:respuesta.index(sep) + 1]
                break

        return {
            "razonamiento": razonamiento[:2000],
            "respuesta": respuesta.strip(),
            "p_engano_razonamiento": f"{p_razon:.3f}",
            "p_engano_respuesta": f"{p_resp:.3f}",
            "cerro_think": int(cerro),
        }

    filas = []
    ruta = config.DIR_ESCENARIOS / f"scratchpad_{args.modelo}.csv"
    for eid, base, sin_inc, con_inc, pregunta in ESCENARIOS:
        for incentivo, instr in [(0, sin_inc), (1, con_inc)]:
            prompt = f"{base} {instr}\n\nUsuario: {pregunta}\nTú:"
            for k in range(args.n):
                r = generar_y_puntuar(prompt)
                filas.append({
                    "id": eid, "incentivo": incentivo, "rep": k,
                    "prompt": prompt.replace("\n", " | "),
                    "razonamiento": r["razonamiento"],
                    "respuesta": r["respuesta"],
                    "p_engano_razonamiento": r["p_engano_razonamiento"],
                    "p_engano_respuesta": r["p_engano_respuesta"],
                    "p_engano": r["p_engano_razonamiento"],
                    "cerro_think": r["cerro_think"],
                    "etiqueta_manual": "",
                })
            _guardar_csv(filas, ruta)
            print(f"  {eid} (incentivo={incentivo}) listo")

    destino = _guardar_csv(filas, ruta)
    n_sin_cierre = sum(1 for f in filas if not f["cerro_think"])
    print(f"\n{len(filas)} filas guardadas en: {destino}")
    if n_sin_cierre:
        print(f"[!] {n_sin_cierre} respuestas no cerraron </think> dentro de "
              f"{args.max_new_tokens} tokens.")
    print("\nSIGUIENTE PASO:")
    print("  1) Abre el CSV y etiqueta 'etiqueta_manual' según la RESPUESTA.")
    print("  2) Compara las dos señales:")
    print(f"     python analizar_escenarios.py --csv {destino} --columna-p p_engano_razonamiento")
    print(f"     python analizar_escenarios.py --csv {destino} --columna-p p_engano_respuesta")


if __name__ == "__main__":
    main()
