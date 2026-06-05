#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gzip
import csv
import math
from itertools import permutations
from datetime import datetime

import numpy as np



# Ruta al LHE. Puede ser .lhe o .lhe.gz.
# Revisa cuál existe en:
# /home/jadi/software/MG5_aMC_v2_9_24/COEFPPZMUE/Events/run_01/
RUTA_LHE = "/home/jadi/software/MG5_aMC_v2_9_24/COEFZFOTONMUE/Events/run_02/unweighted_events.lhe"

# Si tu salida está descomprimida como carpeta, usa algo como:
# RUTA_LHE = "/home/jadi/software/MG5_aMC_v2_9_24/COEFPPZMUE/Events/run_01/unweighted_events.lhe/tmp_0_unweighted_events.lhe"

# Carpeta donde se guardan los coeficientes completos A0...A7.
OUTDIR_Ai = "salidacoef_COEFZFOTONMUE_run02_SIGNY_MINUS"

# Carpeta donde se guarda la tabla reducida para Tapia/LHCb.
OUTDIR_TABLACOMP = "SALIDACOMPARACION_COEFZFOTONMUE_run02_SIGNY_MINUS"

# Ventana de masa del par leptónico.
MASS_MIN = 80.0
MASS_MAX = 100.0

# Bins de QT = pT(ll).
QT_BINS = np.array(
    [0, 10, 15, 20, 25, 30, 35, 45, 55, 70, 90, 120, 160, 220, 300, 500],
    dtype=float
)

# Mínimo de eventos para aceptar un bin.
MIN_EVENTS_PER_BIN = 20

# Convención del leptón para definir los ángulos:

LEPTON_FOR_ANGLES = "minus"


ORIENTAR_Z_POR_SIGNO_Y = True

# Si True, descarta eventos con QT casi cero, donde Collins-Soper degenera.
SKIP_DEGENERATE_CS = True
QT_EPS = 1.0e-8


ARCHIVO_DIAGNOSTICO_Ai = os.path.join(OUTDIR_Ai, "diagnostico_coeficientes_Ai.txt")
ARCHIVO_TABLA_Ai_TXT = os.path.join(OUTDIR_Ai, "tabla_coeficientes_Ai.txt")
ARCHIVO_Ai_CSV = os.path.join(OUTDIR_Ai, "coeficientes_Ai.csv")


ARCHIVO_TABLACOMP_TXT = os.path.join(OUTDIR_TABLACOMP, "tabla_comparativa.txt")
ARCHIVO_TABLACOMP_CSV = os.path.join(OUTDIR_TABLACOMP, "tabla_comparativa.csv")

MZ = 91.1876

os.makedirs(OUTDIR_Ai, exist_ok=True)
os.makedirs(OUTDIR_TAPIA, exist_ok=True)



LOG_LINES = []

def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))

def guardar_log():
    with open(ARCHIVO_DIAGNOSTICO_Ai, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
        f.write("\n")




def abrir_lhe(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, "r", errors="ignore")




def cuadrivector(px, py, pz, e):
    return np.array([e, px, py, pz], dtype=float)

def producto_escalar(a, b):
    return a[0]*b[0] - np.dot(a[1:], b[1:])

def norma2(p):
    return producto_escalar(p, p)

def masa_invariante(p):
    return np.sqrt(max(norma2(p), 0.0))

def pt(p):
    return np.hypot(p[1], p[2])

def rapidez(p):
    E = p[0]
    pz = p[3]
    if E <= abs(pz):
        return np.nan
    return 0.5*np.log((E + pz)/(E - pz))

def boost_al_reposo(Q, p):
    """
    Boost de p al sistema de reposo de Q.
    """
    M = masa_invariante(Q)
    if M < 1e-12:
        return p.copy()

    beta = Q[1:] / Q[0]
    beta2 = np.dot(beta, beta)

    if beta2 < 1e-20:
        return p.copy()

    if beta2 >= 1.0:
        return p.copy()

    gamma = 1.0 / np.sqrt(1.0 - beta2)

    E = p[0]
    vec = p[1:]
    bp = np.dot(beta, vec)

    E_new = gamma * (E - bp)
    vec_new = vec + ((gamma - 1.0)*bp/beta2 - gamma*E)*beta

    return np.array([E_new, vec_new[0], vec_new[1], vec_new[2]], dtype=float)


# ============================================================
# COLLINS-SOPER EJES
# ============================================================

def bajar_indice(v):
    return np.array([v[0], -v[1], -v[2], -v[3]], dtype=float)

def signo_perm(perm):
    inv = 0
    for i in range(4):
        for j in range(i + 1, 4):
            if perm[i] > perm[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1

def levi_civita_4d(a, b, c):
    ac = bajar_indice(a)
    bc = bajar_indice(b)
    cc = bajar_indice(c)

    d = np.zeros(4)

    for perm in permutations([0, 1, 2, 3]):
        mu, nu, alpha, beta = perm
        d[mu] += signo_perm(perm) * ac[nu] * bc[alpha] * cc[beta]

    return d

def construir_CS(Q):
    """
    Construye los cuatro-vectores X,Y,Z del frame Collins-Soper.
    Para QT=0, X/Y pueden degenerarse.
    """
    PA = np.array([1.0, 0.0, 0.0,  1.0])
    PB = np.array([1.0, 0.0, 0.0, -1.0])

    QPA = producto_escalar(Q, PA)
    QPB = producto_escalar(Q, PB)
    Q2 = norma2(Q)

    if abs(QPA) < 1e-12 or abs(QPB) < 1e-12:
        return None

    Zt = QPB*PA - QPA*PB
    Xt = Q - (Q2/(2.0*QPA))*PA - (Q2/(2.0*QPB))*PB
    Yt = levi_civita_4d(PA, PB, Q)

    nX2 = -norma2(Xt)
    nY2 = -norma2(Yt)
    nZ2 = -norma2(Zt)

    if nX2 <= 1e-20 or nY2 <= 1e-20 or nZ2 <= 1e-20:
        return None

    X = Xt / np.sqrt(nX2)
    Y = Yt / np.sqrt(nY2)
    Z = Zt / np.sqrt(nZ2)

    return X, Y, Z

def ejes_en_reposo(Q, CS):
    """
    Lleva X,Y,Z al reposo de Q, ortonormaliza.
    """
    X, Y, Z = CS

    Xr = boost_al_reposo(Q, X)[1:]
    Yr = boost_al_reposo(Q, Y)[1:]
    Zr = boost_al_reposo(Q, Z)[1:]

    def unit(v):
        n = np.linalg.norm(v)
        if n < 1e-12:
            return None
        return v / n

    ex = unit(Xr)
    if ex is None:
        return None

    Yr = Yr - np.dot(Yr, ex)*ex
    ey = unit(Yr)
    if ey is None:
        return None

    ez = np.cross(ex, ey)
    ez = unit(ez)
    if ez is None:
        return None

    # Orientación consistente con Zr si es posible.
    ztest = unit(Zr)
    if ztest is not None and np.dot(ez, ztest) < 0:
        ey = -ey
        ez = -ez

    return ex, ey, ez

def angulos_collins_soper(lminus, lplus):
    """
        cos(theta*), phi*, QT, Mll, yll
    """
    Q = lminus + lplus
    Mll = masa_invariante(Q)
    QT = pt(Q)
    yll = rapidez(Q)

    if Mll <= 0:
        return None

    if QT < QT_EPS and SKIP_DEGENERATE_CS:
        return None

    CS = construir_CS(Q)

    if CS is None:
        return None

    ejes = ejes_en_reposo(Q, CS)
    if ejes is None:
        return None

    ex, ey, ez = ejes

    lep = lminus if LEPTON_FOR_ANGLES == "minus" else lplus
    lep_rest = boost_al_reposo(Q, lep)

    v = lep_rest[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return None

    v = v / nv

    x = np.dot(v, ex)
    y = np.dot(v, ey)
    z = np.dot(v, ez)

    cos_theta = np.clip(z, -1.0, 1.0)
    phi = np.arctan2(y, x)

    return cos_theta, phi, QT, Mll, yll


# ============================================================
# LECTURA DEL LHE Y RECONSTRUCCIÓN l--
# ============================================================

def procesar_lhe():
    eventos = []

    particulas = []
    leyendo = False

    n_eventos = 0
    n_con_ll = 0
    n_mumu = 0
    n_ee = 0
    n_masa = 0
    n_cs_ok = 0
    n_cs_fail = 0

    with abrir_lhe(RUTA_LHE) as f:
        for linea in f:
            linea = linea.strip()

            if linea.startswith("<event"):
                leyendo = True
                particulas = []
                continue

            if linea.startswith("</event"):
                leyendo = False
                n_eventos += 1

                mu_minus = []
                mu_plus = []
                e_minus = []
                e_plus = []
                peso = 1.0

                # Header: nexternal, idprup, weight, scale, aqed, aqcd
                if len(particulas) > 0 and len(particulas[0]) == 6:
                    try:
                        peso = float(particulas[0][2])
                    except Exception:
                        peso = 1.0

                for p in particulas:
                    if len(p) < 10:
                        continue

                    try:
                        pid = int(p[0])
                        status = int(p[1])
                        px = float(p[6])
                        py = float(p[7])
                        pz = float(p[8])
                        e = float(p[9])
                    except Exception:
                        continue

                    if status != 1:
                        continue

                    p4 = cuadrivector(px, py, pz, e)

                    # PDG:
                    #   mu- =  13, mu+ = -13
                    #   e-  =  11, e+  = -11
                    if pid == 13:
                        mu_minus.append(p4)
                    elif pid == -13:
                        mu_plus.append(p4)
                    elif pid == 11:
                        e_minus.append(p4)
                    elif pid == -11:
                        e_plus.append(p4)

                # Construimos pares válidos sin mezclar sabores.
                pares = []

                for lm in mu_minus:
                    for lp in mu_plus:
                        pares.append(("mumu", lm, lp))

                for lm in e_minus:
                    for lp in e_plus:
                        pares.append(("ee", lm, lp))

                if len(pares) < 1:
                    continue

                n_con_ll += 1

                # Si hubiera más de un par, usamos el más cercano a MZ.
                mejor = None
                for sabor, lm, lp in pares:
                    Q = lm + lp
                    M = masa_invariante(Q)
                    cand = (abs(M - MZ), sabor, lm, lp, M)
                    if mejor is None or cand[0] < mejor[0]:
                        mejor = cand

                _, sabor, lm, lp, M = mejor

                if sabor == "mumu":
                    n_mumu += 1
                elif sabor == "ee":
                    n_ee += 1

                if not (MASS_MIN <= M <= MASS_MAX):
                    continue

                n_masa += 1

                ang = angulos_collins_soper(lm, lp)
                if ang is None:
                    n_cs_fail += 1
                    continue

                cos_theta, phi, QT, Mll, yll = ang

                cos_theta_raw = cos_theta
                signo_y = 1.0
                if ORIENTAR_Z_POR_SIGNO_Y:
                    signo_y = 1.0 if yll >= 0.0 else -1.0
                    cos_theta = signo_y * cos_theta

                n_cs_ok += 1

                eventos.append({
                    "weight": peso,
                    "cos_theta": cos_theta,
                    "cos_theta_raw": cos_theta_raw,
                    "sign_y": signo_y,
                    "phi": phi,
                    "QT": QT,
                    "Mll": Mll,
                    "yll": yll,
                    "flavor": sabor,
                })

                if n_eventos % 50000 == 0:
                    log(
                        f"Eventos leídos: {n_eventos:8d} | "
                        f"con ll: {n_con_ll:8d} | "
                        f"mumu: {n_mumu:8d} | "
                        f"ee: {n_ee:8d} | "
                        f"masa: {n_masa:8d} | "
                        f"CS ok: {n_cs_ok:8d}"
                    )

                continue

            if leyendo:
                partes = linea.split()

                if not partes:
                    continue

                if len(partes) == 6:
                    try:
                        int(partes[0])
                        float(partes[2])
                        particulas.append(partes)
                        continue
                    except ValueError:
                        pass

                if len(partes) >= 10:
                    try:
                        int(partes[0])
                        int(partes[1])
                        particulas.append(partes)
                    except ValueError:
                        pass

    log("\n" + "=" * 90)
    log("RESUMEN DE LECTURA")
    log("=" * 90)
    log(f"LHE                         : {RUTA_LHE}")
    log(f"Eventos leídos              : {n_eventos}")
    log(f"Eventos con par leptónico   : {n_con_ll}")
    log(f"Eventos con mu- mu+         : {n_mumu}")
    log(f"Eventos con e- e+           : {n_ee}")
    log(f"Eventos en ventana de masa  : {n_masa}")
    log(f"Eventos con Collins-Soper OK: {n_cs_ok}")
    log(f"Eventos descartados CS      : {n_cs_fail}")
    log(f"Ventana de masa usada       : {MASS_MIN:.1f} < Mll < {MASS_MAX:.1f} GeV")
    log(f"Leptón usado para ángulos   : {LEPTON_FOR_ANGLES}")
    log(f"SKIP_DEGENERATE_CS          : {SKIP_DEGENERATE_CS}")
    log(f"ORIENTAR_Z_POR_SIGNO_Y      : {ORIENTAR_Z_POR_SIGNO_Y}")
    log("Canales aceptados           : mu- mu+ y e- e+")
    log("Nota                        : no se mezclan sabores e/mu")

    if n_masa > 0 and n_cs_ok == 0:
        log("\nADVERTENCIA IMPORTANTE:")
        log("No se pudo construir Collins-Soper para eventos útiles.")
        log("Esto suele ocurrir si el proceso no tiene recoil transversal suficiente.")
        log("Para extraer A_i(QT) usa, por ejemplo:")
        log("    generate p p > mu- mu+ j QED=2 QCD=1")

    return eventos


# ============================================================
# EXTRACCIÓN DE COEFICIENTES Ai POR MOMENTOS
# ============================================================

def media_ponderada(x, w):
    sw = np.sum(w)
    if sw == 0:
        return np.nan
    return np.sum(w*x)/sw

def error_media_ponderada(x, w):
    """
    Error estadístico aproximado de la media ponderada.
    """
    sw = np.sum(w)
    if sw == 0:
        return np.nan

    mean = media_ponderada(x, w)
    var = np.sum(w*w*(x - mean)**2)/(sw*sw)
    return np.sqrt(max(var, 0.0))

def calcular_Ai(cos_theta, phi, weights):
    """
    Método de momentos para:

    dσ/dΩ ∝
      (1 + cos²θ)
      + A0/2 (1 - 3cos²θ)
      + A1 sin(2θ) cosφ
      + A2/2 sin²θ cos(2φ)
      + A3 sinθ cosφ
      + A4 cosθ
      + A5 sin²θ sin(2φ)
      + A6 sin(2θ) sinφ
      + A7 sinθ sinφ

    Momentos:
      A0 = 4 - 10 <cos²θ>
      A1 = 5 <sin2θ cosφ>
      A2 = 10 <sin²θ cos2φ>
      A3 = 4 <sinθ cosφ>
      A4 = 4 <cosθ>
      A5 = 5 <sin²θ sin2φ>
      A6 = 5 <sin2θ sinφ>
      A7 = 4 <sinθ sinφ>
    """
    c = np.asarray(cos_theta, dtype=float)
    ph = np.asarray(phi, dtype=float)
    w = np.asarray(weights, dtype=float)

    s = np.sqrt(np.maximum(0.0, 1.0 - c*c))
    sin2t = 2.0*s*c
    sin2 = s*s

    momentos = {
        "A0": c*c,
        "A1": sin2t*np.cos(ph),
        "A2": sin2*np.cos(2.0*ph),
        "A3": s*np.cos(ph),
        "A4": c,
        "A5": sin2*np.sin(2.0*ph),
        "A6": sin2t*np.sin(ph),
        "A7": s*np.sin(ph),
    }

    factores = {
        "A0": -10.0,
        "A1": 5.0,
        "A2": 10.0,
        "A3": 4.0,
        "A4": 4.0,
        "A5": 5.0,
        "A6": 5.0,
        "A7": 4.0,
    }

    offsets = {"A0": 4.0}

    resultados = {}

    for name in [f"A{i}" for i in range(8)]:
        m = momentos[name]
        mean = media_ponderada(m, w)
        err_mean = error_media_ponderada(m, w)

        val = offsets.get(name, 0.0) + factores[name]*mean
        err = abs(factores[name])*err_mean

        resultados[name] = val
        resultados[name + "_err"] = err

    # tabla comparativa
    resultados["Sx_from_A3"] = resultados["A3"]/4.0
    resultados["Sy_from_A7"] = resultados["A7"]/4.0
    resultados["Sz_from_A4"] = resultados["A4"]/4.0

    resultados["Sx_err"] = resultados["A3_err"]/4.0
    resultados["Sy_err"] = resultados["A7_err"]/4.0
    resultados["Sz_err"] = resultados["A4_err"]/4.0

    sw = np.sum(w)
    sw2 = np.sum(w*w)
    resultados["N_eff"] = sw*sw/sw2 if sw2 > 0 else 0.0

    return resultados

def analizar_por_bins(eventos):
    filas = []

    if len(eventos) == 0:
        return filas

    QT = np.array([e["QT"] for e in eventos], dtype=float)
    Mll = np.array([e["Mll"] for e in eventos], dtype=float)
    yll = np.array([e["yll"] for e in eventos], dtype=float)
    cos_theta = np.array([e["cos_theta"] for e in eventos], dtype=float)
    phi = np.array([e["phi"] for e in eventos], dtype=float)
    w = np.array([e["weight"] for e in eventos], dtype=float)

    # Bin total
    res_total = calcular_Ai(cos_theta, phi, w)
    filas.append({
        "bin": "TOTAL",
        "QT_low": np.nan,
        "QT_high": np.nan,
        "QT_mean": media_ponderada(QT, w),
        "Mll_mean": media_ponderada(Mll, w),
        "yll_mean": media_ponderada(yll, w),
        "N": len(QT),
        **res_total,
    })

    # Bins en QT
    for i in range(len(QT_BINS) - 1):
        lo = QT_BINS[i]
        hi = QT_BINS[i+1]

        mask = (QT >= lo) & (QT < hi)
        n = int(np.sum(mask))

        if n < MIN_EVENTS_PER_BIN:
            continue

        res = calcular_Ai(cos_theta[mask], phi[mask], w[mask])

        filas.append({
            "bin": f"{lo:.1f}_{hi:.1f}",
            "QT_low": lo,
            "QT_high": hi,
            "QT_mean": media_ponderada(QT[mask], w[mask]),
            "Mll_mean": media_ponderada(Mll[mask], w[mask]),
            "yll_mean": media_ponderada(yll[mask], w[mask]),
            "N": n,
            **res,
        })

    return filas


# ============================================================
# SALIDAS  A0...A7
# ============================================================

def guardar_tabla_Ai(filas):
    if len(filas) == 0:
        log("\nNo hay filas para guardar.")
        return

    columnas = [
        "bin", "QT_low", "QT_high", "QT_mean", "Mll_mean", "yll_mean", "N", "N_eff",
        "A0", "A0_err", "A1", "A1_err", "A2", "A2_err", "A3", "A3_err",
        "A4", "A4_err", "A5", "A5_err", "A6", "A6_err", "A7", "A7_err",
        "Sx_from_A3", "Sx_err", "Sy_from_A7", "Sy_err", "Sz_from_A4", "Sz_err",
    ]

    with open(ARCHIVO_Ai_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        for fila in filas:
            writer.writerow(fila)

    with open(ARCHIVO_TABLA_Ai_TXT, "w", encoding="utf-8") as f:
        f.write("="*140 + "\n")
        f.write("TABLA DE COEFICIENTES ANGULARES A0...A7\n")
        f.write("="*140 + "\n")
        f.write(f"LHE: {RUTA_LHE}\n")
        f.write(f"Masa: {MASS_MIN:.1f} < Mll < {MASS_MAX:.1f} GeV\n")
        f.write(f"Leptón usado para ángulos: {LEPTON_FOR_ANGLES}\n"
        f"Orientar z por sign(yZ): {ORIENTAR_Z_POR_SIGNO_Y}\n")
        f.write("Canales aceptados: mu- mu+ y e- e+\n\n")

        f.write(
            f"{'bin':>14s} {'N':>8s} {'QTmean':>10s} "
            f"{'A0':>11s} {'err':>9s} {'A1':>11s} {'err':>9s} "
            f"{'A2':>11s} {'err':>9s} {'A3':>11s} {'err':>9s} "
            f"{'A4':>11s} {'err':>9s} {'A5':>11s} {'err':>9s} "
            f"{'A6':>11s} {'err':>9s} {'A7':>11s} {'err':>9s}\n"
        )
        f.write("-"*180 + "\n")

        for r in filas:
            f.write(
                f"{r['bin']:>14s} {int(r['N']):8d} {r['QT_mean']:10.4f} "
                f"{r['A0']:11.6f} {r['A0_err']:9.6f} "
                f"{r['A1']:11.6f} {r['A1_err']:9.6f} "
                f"{r['A2']:11.6f} {r['A2_err']:9.6f} "
                f"{r['A3']:11.6f} {r['A3_err']:9.6f} "
                f"{r['A4']:11.6f} {r['A4_err']:9.6f} "
                f"{r['A5']:11.6f} {r['A5_err']:9.6f} "
                f"{r['A6']:11.6f} {r['A6_err']:9.6f} "
                f"{r['A7']:11.6f} {r['A7_err']:9.6f}\n"
            )

        f.write("\n")
        f.write("="*140 + "\n")
        f.write("RELACIÓN CON TAPIA\n")
        f.write("="*140 + "\n")
        f.write("Sx = A3/4, Sy = A7/4, Sz = A4/4\n\n")

        f.write(
            f"{'bin':>14s} {'Sx=A3/4':>14s} {'err':>10s} "
            f"{'Sy=A7/4':>14s} {'err':>10s} "
            f"{'Sz=A4/4':>14s} {'err':>10s}\n"
        )
        f.write("-"*90 + "\n")

        for r in filas:
            f.write(
                f"{r['bin']:>14s} "
                f"{r['Sx_from_A3']:14.6f} {r['Sx_err']:10.6f} "
                f"{r['Sy_from_A7']:14.6f} {r['Sy_err']:10.6f} "
                f"{r['Sz_from_A4']:14.6f} {r['Sz_err']:10.6f}\n"
            )

    log("\n" + "="*90)
    log("SALIDAS A_i GUARDADAS")
    log("="*90)
    log(f"Diagnóstico TXT : {ARCHIVO_DIAGNOSTICO_Ai}")
    log(f"Tabla TXT       : {ARCHIVO_TABLA_Ai_TXT}")
    log(f"CSV             : {ARCHIVO_Ai_CSV}")


def imprimir_resumen_Ai(filas):
    if len(filas) == 0:
        log("\nNo se obtuvieron coeficientes.")
        return

    log("\n" + "="*140)
    log("COEFICIENTES ANGULARES EXTRAÍDOS")
    log("="*140)
    log(
        f"{'bin':>14s} {'N':>8s} {'QTmean':>10s} "
        f"{'A0':>11s} {'A1':>11s} {'A2':>11s} {'A3':>11s} "
        f"{'A4':>11s} {'A5':>11s} {'A6':>11s} {'A7':>11s}"
    )
    log("-"*140)

    for r in filas:
        log(
            f"{r['bin']:>14s} {int(r['N']):8d} {r['QT_mean']:10.4f} "
            f"{r['A0']:11.6f} {r['A1']:11.6f} {r['A2']:11.6f} {r['A3']:11.6f} "
            f"{r['A4']:11.6f} {r['A5']:11.6f} {r['A6']:11.6f} {r['A7']:11.6f}"
        )

    log("-"*140)
    log("Relación con Tapia: Sx=A3/4, Sy=A7/4, Sz=A4/4")


# ============================================================
# TABLA REDUCIDA 
# ============================================================

def fmt(x, nd=6):
    if x is None:
        return "nan"
    try:
        if math.isnan(x):
            return "nan"
    except Exception:
        pass
    return f"{x:.{nd}f}"

def construir_filas_tapia(filas_Ai, incluir_total=False):
    filas_tapia = []

    for r in filas_Ai:
        if not incluir_total and str(r.get("bin", "")).upper().startswith("TOTAL"):
            continue

        A0 = r["A0"]
        A2 = r["A2"]
        A3 = r["A3"]
        A4 = r["A4"]
        A7 = r["A7"]

        A0err = r.get("A0_err", np.nan)
        A2err = r.get("A2_err", np.nan)
        A3err = r.get("A3_err", np.nan)
        A4err = r.get("A4_err", np.nan)
        A7err = r.get("A7_err", np.nan)

        fila = {
            "bin": r.get("bin", ""),
            "QTmean": r["QT_mean"],
            "A0": A0,
            "A0_err": A0err,
            "A2": A2,
            "A2_err": A2err,
            "A0_minus_A2": A0 - A2,
            "A0_minus_A2_err": math.sqrt(A0err**2 + A2err**2),
            "A3": A3,
            "A3_err": A3err,
            "A4": A4,
            "A4_err": A4err,
            "A7": A7,
            "A7_err": A7err,
            "Sx": A3/4.0,
            "Sx_err": A3err/4.0,
            "Sy": A7/4.0,
            "Sy_err": A7err/4.0,
            "Sz": A4/4.0,
            "Sz_err": A4err/4.0,
        }

        filas_tapia.append(fila)

    return filas_tapia

def guardar_tapia_csv(filas, path):
    columnas = [
        "bin", "QTmean",
        "A0", "A0_err",
        "A2", "A2_err",
        "A0_minus_A2", "A0_minus_A2_err",
        "A3", "A3_err",
        "A4", "A4_err",
        "A7", "A7_err",
        "Sx", "Sx_err",
        "Sy", "Sy_err",
        "Sz", "Sz_err",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        for r in filas:
            writer.writerow(r)

def guardar_tapia_txt(filas, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 150 + "\n")
        f.write("TABLA COMPARATIVA PARA LHCb / TAPIA\n")
        f.write("=" * 150 + "\n\n")
        f.write(f"Entrada directa: {RUTA_LHE}\n")
        f.write(f"CSV completo A_i: {ARCHIVO_Ai_CSV}\n\n")
        f.write(f"Convención usada: {LEPTON_FOR_ANGLES.upper()}\n"
        f"Orientar z por sign(yZ): {ORIENTAR_Z_POR_SIGNO_Y}\n")
        f.write("Canales aceptados: mu- mu+ y e- e+\n")
        f.write("Relación con Tapia:\n")
        f.write("    Sx = A3/4\n")
        f.write("    Sy = A7/4\n")
        f.write("    Sz = A4/4\n\n")
        f.write("La columna A0-A2 sirve para comparar con la relación de Lam-Tung.\n\n")

        f.write(
            f"{'bin':>14s} {'QTmean':>10s} "
            f"{'A0':>11s} {'A2':>11s} {'A0-A2':>11s} "
            f"{'A3':>11s} {'A4':>11s} {'A7':>11s} "
            f"{'Sx':>11s} {'Sy':>11s} {'Sz':>11s}\n"
        )
        f.write("-" * 150 + "\n")

        for r in filas:
            f.write(
                f"{r['bin']:>14s} {fmt(r['QTmean'],4):>10s} "
                f"{fmt(r['A0']):>11s} {fmt(r['A2']):>11s} {fmt(r['A0_minus_A2']):>11s} "
                f"{fmt(r['A3']):>11s} {fmt(r['A4']):>11s} {fmt(r['A7']):>11s} "
                f"{fmt(r['Sx']):>11s} {fmt(r['Sy']):>11s} {fmt(r['Sz']):>11s}\n"
            )

        f.write("-" * 150 + "\n\n")

        f.write("Interpretación rápida:\n")
        f.write("  - A0 y A2 describen la estructura cuadrupolar principal de la distribución angular.\n")
        f.write("  - A0-A2 prueba la relación de Lam-Tung.\n")
        f.write("  - A3, A4 y A7 se traducen directamente al vector de spin usado por Tapia.\n")
        f.write("  - Para el campo tipo vórtice se usan principalmente Sx(QT) y Sy(QT).\n")

def imprimir_tapia(filas):
    log("\n" + "=" * 150)
    log("TABLA COMPARATIVA PARA LHCb / TAPIA")
    log("=" * 150)
    log(
        f"{'bin':>14s} {'QTmean':>10s} "
        f"{'A0':>11s} {'A2':>11s} {'A0-A2':>11s} "
        f"{'A3':>11s} {'A4':>11s} {'A7':>11s} "
        f"{'Sx':>11s} {'Sy':>11s} {'Sz':>11s}"
    )
    log("-" * 150)

    for r in filas:
        log(
            f"{r['bin']:>14s} {fmt(r['QTmean'],4):>10s} "
            f"{fmt(r['A0']):>11s} {fmt(r['A2']):>11s} {fmt(r['A0_minus_A2']):>11s} "
            f"{fmt(r['A3']):>11s} {fmt(r['A4']):>11s} {fmt(r['A7']):>11s} "
            f"{fmt(r['Sx']):>11s} {fmt(r['Sy']):>11s} {fmt(r['Sz']):>11s}"
        )

    log("-" * 150)

def guardar_tabla_tapia(filas_tapia):
    if len(filas_tapia) == 0:
        log("\nNo hay filas para tabla Tapia/LHCb.")
        return

    imprimir_tapia(filas_tapia)
    guardar_tapia_csv(filas_tapia, ARCHIVO_TAPIA_CSV)
    guardar_tapia_txt(filas_tapia, ARCHIVO_TAPIA_TXT)

    log("\n" + "="*90)
    log("SALIDAS TAPIA/LHCb GUARDADAS")
    log("="*90)
    log(f"Tabla TXT : {ARCHIVO_TAPIA_TXT}")
    log(f"CSV       : {ARCHIVO_TAPIA_CSV}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    log("="*90)
    log("EXTRACCIÓN DE A_i + TABLA COMPARATIVA TAPIA/LHCb")
    log("="*90)
    log(f"Fecha/hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"LHE: {RUTA_LHE}")
    log(f"OUTDIR A_i: {OUTDIR_Ai}")
    log(f"OUTDIR Tapia: {OUTDIR_TAPIA}")
    log(f"Ventana de masa: {MASS_MIN:.1f} < Mll < {MASS_MAX:.1f} GeV")
    log(f"Bins QT: {QT_BINS}")
    log(f"Leptón para ángulos: {LEPTON_FOR_ANGLES}")
    log("Canales aceptados: mu- mu+ y e- e+")
    log("Generación: Z/gamma* + jet LO, canales mu- mu+ y e- e+")
    log("No se excluye el fotón: incluye Z, gamma* e interferencia")
    log(f"Orientación z por sign(yZ): {ORIENTAR_Z_POR_SIGNO_Y}")

    eventos = procesar_lhe()
    filas_Ai = analizar_por_bins(eventos)

    imprimir_resumen_Ai(filas_Ai)
    guardar_tabla_Ai(filas_Ai)

    filas_tapia = construir_filas_tapia(filas_Ai, incluir_total=False)
    guardar_tabla_tapia(filas_tapia)

    guardar_log()

    log("\nLISTO.")
