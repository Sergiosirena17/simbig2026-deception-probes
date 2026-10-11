import numpy as np
from sklearn.linear_model import LogisticRegression


def _a_pm1(etiquetas):
    e = np.asarray(etiquetas)
    return np.where(e > 0, 1.0, -1.0)


def centrar_por_topico(activaciones, topicos):
    A = np.asarray(activaciones, dtype=np.float64)
    topicos = np.asarray(topicos)
    A_c = np.empty_like(A)
    medias = {}
    for t in np.unique(topicos):
        mask = topicos == t
        mu = A[mask].mean(axis=0)
        medias[t] = mu
        A_c[mask] = A[mask] - mu
    return A_c, medias


def direccion_mass_mean(activaciones_centradas, etiquetas):
    A = np.asarray(activaciones_centradas, dtype=np.float64)
    y = np.asarray(etiquetas)
    t_mm = A[y == 1].mean(axis=0) - A[y == 0].mean(axis=0)
    return t_mm


def estimar_tG_tP(activaciones, etiquetas_verdad, polaridades, topicos):
    A_c, medias = centrar_por_topico(activaciones, topicos)
    tau = _a_pm1(etiquetas_verdad)
    p = _a_pm1(polaridades)
    X = np.column_stack([tau, tau * p])
    B, *_ = np.linalg.lstsq(X, A_c, rcond=None)
    t_G, t_P = B[0], B[1]
    return t_G, t_P, medias


def direccion_polaridad(activaciones, polaridades):
    A = np.asarray(activaciones, dtype=np.float64)
    p01 = (np.asarray(polaridades) > 0).astype(int)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(A, p01)
    d = lr.coef_[0]
    return d / (np.linalg.norm(d) + 1e-12)


def proyectar(activaciones, *direcciones):
    A = np.asarray(activaciones, dtype=np.float64)
    cols = [A @ (d / (np.linalg.norm(d) + 1e-12)) for d in direcciones]
    return np.column_stack(cols)


def razon_fisher_por_capa(activaciones_por_capa, etiquetas):
    y = np.asarray(etiquetas)
    ratios = {}
    for capa, A in activaciones_por_capa.items():
        A = np.asarray(A, dtype=np.float64)
        media_global = A.mean(axis=0)
        entre = np.zeros(A.shape[1])
        intra = np.zeros(A.shape[1])
        N = len(A)
        for c in (0, 1):
            Ac = A[y == c]
            if len(Ac) == 0:
                continue
            media_c = Ac.mean(axis=0)
            entre += len(Ac) * (media_c - media_global) ** 2
            intra += ((Ac - media_c) ** 2).sum(axis=0)
        entre /= N
        intra /= N
        ratios[capa] = float((entre / (intra + 1e-8)).mean())
    return ratios


class SondaVerdad:

    def __init__(self, metodo="ttpd"):
        self.metodo = metodo
        self.t_G = None
        self.t_P = None
        self.d_pol = None
        self.t_MM = None
        self.clf = None

    def entrenar(self, activaciones, etiquetas, polaridades, topicos):
        A = np.asarray(activaciones, dtype=np.float64)
        y = np.asarray(etiquetas)
        A_c, _ = centrar_por_topico(A, topicos)

        if self.metodo == "lr":
            self.clf = LogisticRegression(max_iter=2000, C=1.0).fit(A, y)
            return self

        if self.metodo in ("tg", "ttpd"):
            self.t_G, self.t_P, _ = estimar_tG_tP(A, y, polaridades, topicos)
        if self.metodo == "mm":
            self.t_MM = direccion_mass_mean(A_c, y)

        if self.metodo == "tg":
            feats = proyectar(A, self.t_G)
        elif self.metodo == "ttpd":
            self.d_pol = direccion_polaridad(A_c, polaridades)
            feats = proyectar(A, self.t_G, self.d_pol)
        elif self.metodo == "mm":
            feats = proyectar(A, self.t_MM)
        else:
            raise ValueError(self.metodo)

        self.clf = LogisticRegression(max_iter=2000, C=1.0).fit(feats, y)
        return self

    def _features(self, activaciones):
        A = np.asarray(activaciones, dtype=np.float64)
        if A.ndim == 1:
            A = A[None, :]
        if self.metodo == "lr":
            return A
        if self.metodo == "tg":
            return proyectar(A, self.t_G)
        if self.metodo == "ttpd":
            return proyectar(A, self.t_G, self.d_pol)
        if self.metodo == "mm":
            return proyectar(A, self.t_MM)
        raise ValueError(self.metodo)

    def prob_verdadero(self, activaciones):
        return self.clf.predict_proba(self._features(activaciones))[:, 1]

    def prob_engano(self, activaciones):
        return 1.0 - self.prob_verdadero(activaciones)
