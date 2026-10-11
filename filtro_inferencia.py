import argparse
import numpy as np
import joblib

import config


INSTRUCCION_HONESTIDAD = (
    "Eres un asistente honesto. Responde con la verdad aunque te perjudique o "
    "contradiga un incentivo. No ocultes ni tergiverses información relevante."
)


class FiltroScheming:
    def __init__(self, ruta_sonda, cuantizar_4bit=None):
        import torch
        self.torch = torch
        paquete = joblib.load(ruta_sonda)
        self.sonda = paquete["sonda"]
        self.capa = paquete["capa"]
        self.nombre_modelo = paquete["modelo"]
        self.info = config.MODELOS[self.nombre_modelo]

        from extraer_activaciones import cargar_modelo
        self.model, self.tokenizer, _, _ = cargar_modelo(
            self.nombre_modelo, cuantizar_4bit=cuantizar_4bit)

        self._vec_steering = self._direccion_steering()

    def _direccion_steering(self):
        s = self.sonda
        d = s.t_G if s.t_G is not None else (s.t_MM if s.t_MM is not None
                                             else s.clf.coef_[0])
        d = np.asarray(d, dtype=np.float32)
        d = d / (np.linalg.norm(d) + 1e-12)
        return self.torch.tensor(d, device=self.model.device, dtype=self.model.dtype)

    def _modulo_capa(self):
        m = self.model
        for parte in self.info["ruta_capas"].split("."):
            m = getattr(m, parte)
        idx = max(0, min(self.capa - 1, len(m) - 1))
        return m[idx]

    def _activacion_respuesta(self, ids_completos):
        torch = self.torch
        with torch.no_grad():
            out = self.model(ids_completos, output_hidden_states=True)
        return out.hidden_states[self.capa][0, -1, :].float().cpu().numpy()

    def _generar(self, system, prompt, max_new_tokens=120):
        torch = self.torch
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        try:
            enc = self.tokenizer.apply_chat_template(
                msgs, return_tensors="pt", add_generation_prompt=True,
                return_dict=True, enable_thinking=False)
        except TypeError:
            enc = self.tokenizer.apply_chat_template(
                msgs, return_tensors="pt", add_generation_prompt=True,
                return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            salida = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.8,
                pad_token_id=self.tokenizer.pad_token_id)
        texto = self.tokenizer.decode(salida[0, n_in:], skip_special_tokens=True)
        if "</think>" in texto:
            texto = texto.split("</think>", 1)[1]
        return texto.strip(), salida

    def puntuar(self, system, prompt, max_new_tokens=120):
        texto, ids = self._generar(system, prompt, max_new_tokens)
        act = self._activacion_respuesta(ids)
        p_eng = float(self.sonda.prob_engano(act)[0])
        return texto, p_eng

    def monitor(self, prompt, system=None, umbral=config.UMBRAL_ENGANO):
        texto, p = self.puntuar(system, prompt)
        return {"respuesta": texto, "p_engano": p, "marcado": p > umbral,
                "modo": "monitor"}

    def regenerar(self, prompt, system=None, umbral=config.UMBRAL_ENGANO,
                  max_reintentos=config.MAX_REINTENTOS):
        intentos = []
        texto, p = self.puntuar(system, prompt)
        intentos.append((texto, p))
        n = 0
        while p > umbral and n < max_reintentos:
            n += 1
            sys_reforzado = (INSTRUCCION_HONESTIDAD if not system
                             else system + " " + INSTRUCCION_HONESTIDAD)
            texto, p = self.puntuar(sys_reforzado, prompt)
            intentos.append((texto, p))
        if p > umbral:
            return {"respuesta": "[Respuesta retenida: el monitor de honestidad "
                                 "detectó posible engaño tras varios intentos.]",
                    "p_engano": p, "marcado": True, "reintentos": n,
                    "intentos": intentos, "modo": "regenerar"}
        return {"respuesta": texto, "p_engano": p, "marcado": False,
                "reintentos": n, "intentos": intentos, "modo": "regenerar"}

    def steering(self, prompt, system=None, alpha=config.ALPHA_STEERING,
                 umbral=config.UMBRAL_ENGANO):
        modulo = self._modulo_capa()
        vec = self._vec_steering

        def hook(mod, entrada, salida):
            if isinstance(salida, tuple):
                hs = salida[0] + alpha * vec
                return (hs,) + tuple(salida[1:])
            return salida + alpha * vec

        h = modulo.register_forward_hook(hook)
        try:
            texto, ids = self._generar(system, prompt)
        finally:
            h.remove()
        act = self._activacion_respuesta(ids)
        p = float(self.sonda.prob_engano(act)[0])
        return {"respuesta": texto, "p_engano": p, "marcado": p > umbral,
                "alpha": alpha, "modo": "steering"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonda", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--modo", default="monitor",
                    choices=["monitor", "regenerar", "steering"])
    ap.add_argument("--umbral", type=float, default=config.UMBRAL_ENGANO)
    ap.add_argument("--alpha", type=float, default=config.ALPHA_STEERING)
    ap.add_argument("--4bit", dest="cuatro_bits", action="store_true")
    args = ap.parse_args()

    f = FiltroScheming(args.sonda, cuantizar_4bit=args.cuatro_bits)
    if args.modo == "monitor":
        r = f.monitor(args.prompt, args.system, args.umbral)
    elif args.modo == "regenerar":
        r = f.regenerar(args.prompt, args.system, args.umbral)
    else:
        r = f.steering(args.prompt, args.system, args.alpha, args.umbral)

    print("\n=== RESULTADO ===")
    print(f"modo       : {r['modo']}")
    print(f"P(engaño)  : {r['p_engano']:.3f}  (umbral {args.umbral})")
    print(f"marcado    : {r['marcado']}")
    if "reintentos" in r:
        print(f"reintentos : {r['reintentos']}")
    print(f"\nrespuesta  : {r['respuesta']}")


if __name__ == "__main__":
    main()
