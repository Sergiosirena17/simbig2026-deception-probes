import argparse
import numpy as np


def evaluar(X, y, grupos, metodo="lr"):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    def nueva_sonda():
        return LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0)

    logo = LeaveOneGroupOut()
    oof = np.full(len(y), np.nan)
    for tr, te in logo.split(X, y, grupos):
        if len(np.unique(y[tr])) < 2:
            oof[te] = float(y[tr][0])
            continue
        sc = StandardScaler().fit(X[tr])
        clf = nueva_sonda().fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    auroc_loso = roc_auc_score(y, oof) if len(np.unique(y)) == 2 else float("nan")

    aurocs_cv = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config_semilla())
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = nueva_sonda().fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aurocs_cv.append(roc_auc_score(y[te], p))
    return auroc_loso, oof, (np.mean(aurocs_cv) if aurocs_cv else float("nan"),
                             np.std(aurocs_cv) if aurocs_cv else 0.0)


def config_semilla():
    try:
        import config
        return config.SEMILLA
    except Exception:
        return 42


def metricas_umbral(y, score, umbral):
    pred = (score > umbral).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1, len(y))
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return acc, prec, rec, f1, (tp, tn, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--umbral", type=float, default=0.5)
    args = ap.parse_args()

    d = np.load(args.acts, allow_pickle=True)
    X, y, grupos = d["X"], d["y"].astype(int), d["grupos"]
    capa = int(d["capa"]) if "capa" in d else "?"
    print(f"Activaciones: {X.shape} | capa {capa}")
    print(f"  mentiras={int(y.sum())} | honestas={int((y==0).sum())} | "
          f"escenarios={len(set(grupos.tolist()))}")
    if y.sum() < 4 or (y == 0).sum() < 4:
        print("[!] Muy pocos ejemplos de una clase; el AUROC será inestable.")

    auroc_loso, oof, (cv_mean, cv_std) = evaluar(X, y, grupos)

    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, oof)
    j = tpr - fpr
    u_opt = float(thr[int(np.argmax(j))])

    acc, prec, rec, f1, conf = metricas_umbral(y, oof, args.umbral)
    acc_o, prec_o, rec_o, f1_o, conf_o = metricas_umbral(y, oof, u_opt)

    print("\n" + "=" * 64)
    print("SONDA DE ENGAÑO (entrenada sobre los escenarios)")
    print("=" * 64)
    print(f"AUROC leave-one-scenario-out : {auroc_loso:.3f}")
    print(f"AUROC CV 5-fold (referencia) : {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"\nEn umbral {args.umbral:.2f}:  exactitud {acc*100:.1f}%  "
          f"precision {prec*100:.1f}%  recall {rec*100:.1f}%  F1 {f1:.3f}")
    print(f"  confusion (TP,TN,FP,FN) = {conf}")
    print(f"Umbral óptimo (Youden) = {u_opt:.3f}  ->  exactitud {acc_o*100:.1f}%  "
          f"recall {rec_o*100:.1f}%  F1 {f1_o:.3f}")
    print(f"  confusion (TP,TN,FP,FN) = {conf_o}")

    print("\nComparación con la sonda de VERDAD aplicada a engaño:")
    print("  - sonda de verdad sobre respuesta  : AUROC ~0.67-0.84 (no detecta; 0% recall)")
    print("  - sonda de verdad sobre razonamiento: AUROC ~0.70")
    print(f"  - sonda de ENGAÑO (esta)           : AUROC {auroc_loso:.3f} (LOSO)")
    if not np.isnan(auroc_loso) and auroc_loso > 0.80:
        print("  -> La dirección de engaño SÍ separa donde la de verdad no. "
              "Resultado positivo para el paper.")
    else:
        print("  -> Mejora limitada; reportar honestamente (probablemente faltan "
              "más ejemplos de mentira).")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import joblib, config
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=4000, class_weight="balanced").fit(sc.transform(X), y)
    ruta = config.DIR_SONDAS / f"sonda_engano_capa{capa}.joblib"
    joblib.dump({"scaler": sc, "clf": clf, "capa": capa, "tipo": "engano"}, ruta)
    print(f"\nSonda de engaño final guardada en: {ruta}")


if __name__ == "__main__":
    main()
