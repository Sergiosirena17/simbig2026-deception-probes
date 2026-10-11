import argparse
import numpy as np
import joblib

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
import sonda_utils as su

METODOS = ["lr", "mm", "tg", "ttpd"]


def cargar(modelo, capa):
    ruta = config.DIR_ACTIVACIONES / f"acts_{modelo}_capa{capa}.npy"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Corre primero:\n"
            f"  python extraer_activaciones.py --modelo {modelo} --capa {capa}")
    A = np.load(ruta)
    y = np.load(config.DIR_ACTIVACIONES / "etiquetas.npy")
    p = np.load(config.DIR_ACTIVACIONES / "polaridades.npy")
    t = np.load(config.DIR_ACTIVACIONES / "topicos.npy", allow_pickle=True)
    return A, y, p, t


def eval_cv_aleatoria(A, y, p, t, metodo, n_splits=5, semilla=config.SEMILLA):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    accs, aurocs = [], []
    for tr, te in skf.split(A, y):
        s = su.SondaVerdad(metodo).entrenar(A[tr], y[tr], p[tr], t[tr])
        pv = s.prob_verdadero(A[te])
        accs.append(np.mean((pv > 0.5).astype(int) == y[te]))
        aurocs.append(roc_auc_score(y[te], pv))
    return np.mean(accs), np.std(accs), np.mean(aurocs), np.std(aurocs)


def eval_leave_one_topic(A, y, p, t, metodo):
    accs, aurocs = [], []
    for topico in np.unique(t):
        te = t == topico
        tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        s = su.SondaVerdad(metodo).entrenar(A[tr], y[tr], p[tr], t[tr])
        pv = s.prob_verdadero(A[te])
        accs.append(np.mean((pv > 0.5).astype(int) == y[te]))
        aurocs.append(roc_auc_score(y[te], pv))
    return np.mean(accs), np.std(accs), np.mean(aurocs), np.std(aurocs)


def barrido_de_capas(modelo):
    ruta = config.DIR_ACTIVACIONES / f"acts_TODAS_{modelo}.npz"
    if not ruta.exists():
        return None
    datos = np.load(ruta)
    y = np.load(config.DIR_ACTIVACIONES / "etiquetas.npy")
    acts_por_capa = {int(k.split("_")[1]): datos[k] for k in datos.files}
    ratios = su.razon_fisher_por_capa(acts_por_capa, y)
    mejor = max(ratios, key=ratios.get)
    print("\nBarrido de capas (razón varianza entre/intra clase):")
    for L in sorted(ratios):
        marca = "  <== mejor" if L == mejor else ""
        print(f"  capa {L:2d}: {ratios[L]:.4f}{marca}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = sorted(ratios)
        plt.figure(figsize=(7, 4))
        plt.plot(xs, [ratios[L] for L in xs], marker="o")
        plt.axvline(mejor, ls="--", color="red", label=f"best = layer {mejor}")
        plt.xlabel("Layer"); plt.ylabel("Inter-class / intra-class variance ratio")
        plt.title(f"True/false separation by layer — {modelo}")
        plt.legend(); plt.tight_layout()
        fig = config.DIR_FIGURAS / f"barrido_capas_{modelo}.png"
        plt.savefig(fig, dpi=150)
        print(f"  figura guardada: {fig}")
    except Exception as e:
        print(f"  (sin gráfica: {e})")
    return mejor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=config.MODELO_DEFECTO, choices=list(config.MODELOS))
    ap.add_argument("--capa", type=int, required=True)
    ap.add_argument("--metodo-final", default="ttpd", choices=METODOS)
    args = ap.parse_args()

    A, y, p, t = cargar(args.modelo, args.capa)
    print(f"Activaciones: {A.shape} | verdaderos={int(y.sum())}/{len(y)} | "
          f"tópicos={sorted(set(t.tolist()))}")

    print("\n" + "=" * 64)
    print(f"{'método':6s} | {'CV-acc':>14s} | {'CV-AUROC':>14s} | "
          f"{'LOTO-acc':>14s} | {'LOTO-AUROC':>14s}")
    print("-" * 64)
    for m in METODOS:
        ca, cas, cau, caus = eval_cv_aleatoria(A, y, p, t, m)
        la, las, lau, laus = eval_leave_one_topic(A, y, p, t, m)
        print(f"{m:6s} | {ca*100:5.1f}±{cas*100:3.1f}%   | {cau:.3f}±{caus:.3f}  | "
              f"{la*100:5.1f}±{las*100:3.1f}%   | {lau:.3f}±{laus:.3f}")
    print("=" * 64)
    print("LOTO = leave-one-topic-out (generalización a tópicos no vistos, el test fuerte).")

    barrido_de_capas(args.modelo)

    sonda = su.SondaVerdad(args.metodo_final).entrenar(A, y, p, t)
    paquete = {
        "sonda": sonda,
        "modelo": args.modelo,
        "hf_id": config.MODELOS[args.modelo]["hf_id"],
        "capa": args.capa,
        "dim": int(A.shape[1]),
        "metodo": args.metodo_final,
    }
    ruta = config.DIR_SONDAS / f"sonda_{args.modelo}_capa{args.capa}_{args.metodo_final}.joblib"
    joblib.dump(paquete, ruta)
    print(f"\nSonda final ({args.metodo_final}) guardada en: {ruta}")


if __name__ == "__main__":
    main()
