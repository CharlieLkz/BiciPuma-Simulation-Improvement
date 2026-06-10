"""
PUMABICIS — Escenario 2: Sistema 100% Autónomo
3 Kioscos RFID/QR + Bloqueo por excepción (P=5%)
UNAM · Ingeniería en Telecomunicaciones · Teoría de Colas
"""

import simpy, random, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, HRFlowable, PageBreak)

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════
INTER_LLEGADAS_MIN = [5, 3, 7, 5, 10, 3, 7, 5, 3, 7, 10]
PERSONAS_POR_TREN  = [4, 7, 1, 3, 8, 4, 2, 5, 4, 3, 1, 8]

LAMBDA_TRENES_S = 1 / (np.mean(INTER_LLEGADAS_MIN) * 60)
MEDIA_LOTE      = np.mean(PERSONAS_POR_TREN)
LAMBDA_EFF_S    = LAMBDA_TRENES_S * MEDIA_LOTE

N_KIOSCOS       = 3
SRV_MIN_S       = 10
SRV_MAX_S       = 20
P_EXCEPCION     = 0.05
BLOQUEO_MIN_S   = 180
BLOQUEO_MAX_S   = 300
PATIENCE_MEAN_S = 300
SIM_TIME_S      = 7200
RANDOM_SEED     = 42
N_REPLICAS      = 10
CARPETA         = "reporte_scen2"
os.makedirs(CARPETA, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# ENTIDADES
# ══════════════════════════════════════════════════════════════════════════════
class Alumno:
    _cnt = 0
    def __init__(self, t_llegada, paciencia, num_tren, pos_en_lote):
        Alumno._cnt += 1
        self.id            = Alumno._cnt
        self.t_llegada     = t_llegada
        self.paciencia     = paciencia
        self.num_tren      = num_tren       # qué tren lo trajo
        self.pos_en_lote   = pos_en_lote    # posición dentro del lote
        self.t_inicio_srv  = None
        self.t_salida      = None
        self.atendido      = False
        self.abandono      = False
        self.fue_excepcion = False

    @property
    def t_espera(self):
        return (self.t_inicio_srv - self.t_llegada) if self.t_inicio_srv else 0.0
    @property
    def t_sistema(self):
        return (self.t_salida - self.t_llegada) if (self.t_salida and self.atendido) else None

class Metricas:
    def __init__(self):
        self.alumnos   = []
        self.log_cola  = []
        self.log_kiosc = []
        self.bloqueos  = []
        self.trenes    = []   # (t_llegada, lote_size, num_tren)

    def registrar(self, a):         self.alumnos.append(a)
    def log_lq(self, t, lq):        self.log_cola.append((t, lq))
    def log_activos(self, t, n):    self.log_kiosc.append((t, n))
    def reg_bloqueo(self, kid, ti, tf):
        self.bloqueos.append({"kiosco": kid, "inicio": ti, "fin": tf, "dur": tf-ti})
    def reg_tren(self, t, lote, num):
        self.trenes.append({"t": t, "lote": lote, "num": num})

    @property
    def total(self):       return len(self.alumnos)
    @property
    def atendidos(self):   return sum(1 for a in self.alumnos if a.atendido)
    @property
    def abandonos(self):   return sum(1 for a in self.alumnos if a.abandono)
    @property
    def excepciones(self): return sum(1 for a in self.alumnos if a.fue_excepcion)
    @property
    def p_abandono(self):  return self.abandonos / self.total if self.total else 0.0
    @property
    def wq_prom(self):
        t = [a.t_espera for a in self.alumnos if a.atendido]
        return np.mean(t) if t else 0.0
    @property
    def wq_max(self):
        t = [a.t_espera for a in self.alumnos if a.atendido]
        return max(t) if t else 0.0
    @property
    def w_prom(self):
        t = [a.t_sistema for a in self.alumnos if a.t_sistema]
        return np.mean(t) if t else 0.0
    @property
    def lq_prom(self):
        if len(self.log_cola) < 2: return 0.0
        ts  = [x[0] for x in self.log_cola]
        lqs = [x[1] for x in self.log_cola]
        _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
        return _trapz(lqs, ts) / (ts[-1] - ts[0] + 1e-9)
    @property
    def lq_max(self): return max((x[1] for x in self.log_cola), default=0)
    def throughput(self, st): return self.atendidos / (st / 3600)
    def tiempo_bloqueado(self): return sum(b["dur"] for b in self.bloqueos)
    def little_error(self):
        lq_l = LAMBDA_EFF_S * self.wq_prom
        lq_e = self.lq_prom
        return abs(lq_l - lq_e) / (lq_e + 1e-9) * 100

# ══════════════════════════════════════════════════════════════════════════════
# RECURSO KIOSCO
# ══════════════════════════════════════════════════════════════════════════════
class KioscoPool:
    def __init__(self, env, n, metricas):
        self.env      = env
        self.resource = simpy.Resource(env, capacity=n)
        self.n        = n
        self.metricas = metricas
        self._bloq    = 0

    def bloquear(self, kid, dur):
        self._bloq += 1
        ti = self.env.now
        self.metricas.log_activos(self.env.now, self.n - self._bloq)
        yield self.env.timeout(dur)
        self._bloq -= 1
        self.metricas.reg_bloqueo(kid, ti, self.env.now)
        self.metricas.log_activos(self.env.now, self.n - self._bloq)

# ══════════════════════════════════════════════════════════════════════════════
# PROCESOS SIMPY
# ══════════════════════════════════════════════════════════════════════════════
def proceso_alumno(env, alumno, pool, metricas, rng):
    metricas.log_lq(env.now, len(pool.resource.queue))
    with pool.resource.request() as req:
        resultado = yield req | env.timeout(alumno.paciencia)
        if req not in resultado:
            alumno.abandono = True
            alumno.t_salida = env.now
            metricas.registrar(alumno)
            metricas.log_lq(env.now, len(pool.resource.queue))
            return
        alumno.t_inicio_srv = env.now
        metricas.log_lq(env.now, len(pool.resource.queue))
        if rng.random() < P_EXCEPCION:
            alumno.fue_excepcion = True
            dur_blq = rng.uniform(BLOQUEO_MIN_S, BLOQUEO_MAX_S)
            kid = rng.randint(1, N_KIOSCOS)
            env.process(pool.bloquear(kid, dur_blq))
            yield env.timeout(dur_blq)
            alumno.abandono = True
            alumno.t_salida = env.now
            metricas.registrar(alumno)
            metricas.log_lq(env.now, len(pool.resource.queue))
        else:
            yield env.timeout(rng.uniform(SRV_MIN_S, SRV_MAX_S))
            alumno.atendido = True
            alumno.t_salida = env.now
            metricas.registrar(alumno)
            metricas.log_lq(env.now, len(pool.resource.queue))

def generador_trenes(env, pool, metricas, rng):
    num_tren = 0
    while True:
        yield env.timeout(rng.expovariate(LAMBDA_TRENES_S))
        num_tren += 1
        lote = rng.randint(1, 8)
        metricas.reg_tren(env.now, lote, num_tren)
        for pos in range(lote):
            pac = rng.expovariate(1 / PATIENCE_MEAN_S)
            a   = Alumno(env.now, pac, num_tren, pos + 1)
            env.process(proceso_alumno(env, a, pool, metricas, rng))

def correr_simulacion(semilla=RANDOM_SEED):
    Alumno._cnt = 0
    rng      = random.Random(semilla)
    env      = simpy.Environment()
    m        = Metricas()
    pool     = KioscoPool(env, N_KIOSCOS, m)
    env.process(generador_trenes(env, pool, m, rng))
    env.run(until=SIM_TIME_S)
    return m

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════
def generar_graficas(m, rep_stats):
    rutas = {}

    # Gráfica 1: Cola en el tiempo con bloqueos marcados
    fig, ax = plt.subplots(figsize=(11, 4))
    ts  = [x[0]/60 for x in m.log_cola]
    lqs = [x[1]    for x in m.log_cola]
    ax.step(ts, lqs, color="#1A5FAF", linewidth=0.7, where='post', alpha=0.85, label="Cola")
    ax.fill_between(ts, lqs, step='post', alpha=0.12, color="#1A5FAF")
    primer_bloqueo = True
    for b in m.bloqueos:
        lbl = "Kiosco bloqueado" if primer_bloqueo else ""
        ax.axvspan(b["inicio"]/60, b["fin"]/60, color="#E06C00", alpha=0.22, label=lbl)
        primer_bloqueo = False
    ax.axhline(m.lq_prom, color="#E24B4A", linestyle="--", linewidth=1.5,
               label=f"Lq promedio = {m.lq_prom:.1f}")
    # Marcar llegadas de trenes
    for tr in m.trenes:
        ax.axvline(tr["t"]/60, color="#1D9E75", alpha=0.3, linewidth=0.8)
    ax.set_title("Cola de espera en el tiempo — zonas naranjas = kiosco bloqueado, "
                 "líneas verdes = llegada de tren", fontsize=10, fontweight='bold')
    ax.set_xlabel("Tiempo transcurrido (minutos)")
    ax.set_ylabel("Alumnos esperando")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, SIM_TIME_S/60)
    plt.tight_layout()
    rutas['cola'] = os.path.join(CARPETA, "graf2_cola.png")
    plt.savefig(rutas['cola'], dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfica 2: Kioscos disponibles en el tiempo
    fig, ax = plt.subplots(figsize=(11, 3.5))
    if m.log_kiosc:
        tk = [x[0]/60 for x in m.log_kiosc]
        nk = [x[1]    for x in m.log_kiosc]
        ax.step(tk, nk, color="#E06C00", linewidth=1.5, where='post', label="Kioscos activos")
        ax.fill_between(tk, nk, step='post', alpha=0.15, color="#E06C00")
    ax.axhline(N_KIOSCOS, color="#1D9E75", linestyle="--", linewidth=1.5,
               label=f"Capacidad total = {N_KIOSCOS}")
    ax.set_ylim(-0.3, N_KIOSCOS + 0.8)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0 (todos bloqueados)", "1 disponible", "2 disponibles", "3 (normal)"])
    ax.set_title("Kioscos disponibles en el tiempo — caídas = bloqueo activo",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Tiempo (minutos)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, SIM_TIME_S/60)
    plt.tight_layout()
    rutas['kioscos'] = os.path.join(CARPETA, "graf2_kioscos.png")
    plt.savefig(rutas['kioscos'], dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfica 3: Histograma esperas
    fig, ax = plt.subplots(figsize=(8, 4))
    esperas = [a.t_espera/60 for a in m.alumnos if a.atendido]
    if esperas:
        ax.hist(esperas, bins=18, color="#1D9E75", edgecolor="white", alpha=0.9)
        ax.axvline(m.wq_prom/60, color="#E24B4A", linestyle="--", linewidth=2,
                   label=f"Promedio = {m.wq_prom/60:.1f} min")
        ax.axvline(5, color="#E06C00", linestyle=":", linewidth=1.5,
                   label="Limite de paciencia (5 min)")
    ax.set_title("Distribucion de tiempos de espera — alumnos atendidos",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Espera en cola (minutos)")
    ax.set_ylabel("Numero de alumnos")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    rutas['esperas'] = os.path.join(CARPETA, "graf2_esperas.png")
    plt.savefig(rutas['esperas'], dpi=150, bbox_inches='tight')
    plt.close()

    # Gráfica 4: Flujo de alumnos + réplicas
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cats = ["Llegaron", "Atendidos", "Abandonaron", "Excepciones"]
    vals = [m.total, m.atendidos, m.abandonos, m.excepciones]
    cols = ["#1A5FAF", "#1D9E75", "#E24B4A", "#E06C00"]
    bars = axes[0].bar(cats, vals, color=cols, width=0.55, edgecolor='white')
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                     str(v), ha='center', fontweight='bold', fontsize=9)
    axes[0].set_title(f"Flujo total\nAbandono: {m.p_abandono:.1%}", fontweight='bold')
    axes[0].set_ylabel("Alumnos"); axes[0].grid(axis='y', alpha=0.3)

    semillas = list(range(N_REPLICAS))
    axes[1].bar(semillas, rep_stats['abandonos'], color="#E24B4A", alpha=0.8, edgecolor='white')
    axes[1].axhline(np.mean(rep_stats['abandonos']), color='black', linestyle='--', linewidth=1.5)
    axes[1].set_title("Abandono por replica (%)", fontweight='bold')
    axes[1].set_xlabel("Replica"); axes[1].set_ylabel("%"); axes[1].grid(alpha=0.3)

    axes[2].bar(semillas, rep_stats['bloqueos'], color="#E06C00", alpha=0.8, edgecolor='white')
    axes[2].axhline(np.mean(rep_stats['bloqueos']), color='black', linestyle='--', linewidth=1.5)
    axes[2].set_title("Bloqueos por replica", fontweight='bold')
    axes[2].set_xlabel("Replica"); axes[2].set_ylabel("N bloqueos"); axes[2].grid(alpha=0.3)

    plt.suptitle("Escenario 2 — Estabilidad en 10 replicas independientes",
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    rutas['replicas'] = os.path.join(CARPETA, "graf2_replicas.png")
    plt.savefig(rutas['replicas'], dpi=150, bbox_inches='tight')
    plt.close()

    return rutas

# ══════════════════════════════════════════════════════════════════════════════
# REPORTE PDF
# ══════════════════════════════════════════════════════════════════════════════
def generar_pdf(m, rep_stats, rutas):
    ruta_pdf = os.path.join(CARPETA, "reporte_pumabicis_scen2.pdf")
    doc      = SimpleDocTemplate(ruta_pdf, pagesize=letter,
                                  leftMargin=0.9*inch, rightMargin=0.9*inch,
                                  topMargin=0.9*inch, bottomMargin=0.9*inch)
    styles = getSampleStyleSheet()
    story  = []

    # Colores
    azul   = colors.HexColor("#1A5FAF")
    verde  = colors.HexColor("#1D9E75")
    rojo   = colors.HexColor("#E24B4A")
    naranj = colors.HexColor("#E06C00")
    gris   = colors.HexColor("#555555")
    azul_c = colors.HexColor("#EEF4FF")
    verd_c = colors.HexColor("#EDFAF4")
    narj_c = colors.HexColor("#FFF3E0")

    st_tit = ParagraphStyle("tit", parent=styles["Title"], fontSize=20,
                             textColor=azul, spaceAfter=4, alignment=TA_CENTER)
    st_sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=11,
                             textColor=gris, spaceAfter=16, alignment=TA_CENTER)
    st_h1  = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13,
                             textColor=azul, spaceBefore=18, spaceAfter=6)
    st_h2  = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11,
                             textColor=verde, spaceBefore=12, spaceAfter=4)
    st_bod = ParagraphStyle("bod", parent=styles["Normal"], fontSize=10,
                             leading=15, spaceAfter=8, alignment=TA_JUSTIFY)
    st_not = ParagraphStyle("not", parent=styles["Normal"], fontSize=9,
                             textColor=gris, leading=13, spaceAfter=6, leftIndent=12)
    st_pie = ParagraphStyle("pie", parent=styles["Normal"], fontSize=8,
                             textColor=colors.HexColor("#999999"), alignment=TA_CENTER)

    def tabla(data, col_w, hdr_color=None, row_colors=None):
        hc = hdr_color or azul
        rc = row_colors or [colors.white, azul_c]
        t  = Table(data, colWidths=col_w)
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), hc),
            ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE',      (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), rc),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ]))
        return t

    # ── PORTADA ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("PUMABICIS", st_tit))
    story.append(Paragraph("Simulacion de Teoria de Colas — Escenario 2: Sistema 100% Autonomo", st_sub))
    story.append(HRFlowable(width="100%", thickness=2, color=azul))
    story.append(Spacer(1, 0.15*inch))
    meta = [
        ["Universidad:",       "UNAM — Facultad de Ingenieria"],
        ["Sistema analizado:", "Cicloestacion Pumabicis — Metro CU"],
        ["Escenario:",         "3 Kioscos RFID/QR sin soporte humano"],
        ["Parametro critico:", "5% de alumnos provoca bloqueo del kiosco de 3 a 5 min"],
        ["Periodo:",           "Hora pico matutina · 05:45 a 07:45 hrs (7200 s)"],
        ["Fecha:",             datetime.now().strftime("%d de %B de %Y · %H:%M hrs")],
    ]
    story.append(tabla(meta, [2.2*inch, 4.5*inch]))
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC")))
    story.append(PageBreak())

    # ── SEC 1: Descripción ───────────────────────────────────────────────────
    story.append(Paragraph("1. Descripcion del Escenario", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    story.append(Paragraph(
        "Este escenario modela una propuesta de <b>automatizacion completa</b> del modulo "
        "Pumabicis mediante la instalacion de <b>3 kioscos RFID/QR</b> que atienden en "
        "10 a 20 segundos sin intervencion humana. Cada kiosco lee la credencial, verifica "
        "el NIP y asigna bicicleta y casco de forma automatica.", st_bod))
    story.append(Paragraph(
        "<b>El problema critico:</b> el 5% de los alumnos genera una excepcion (NIP bloqueado, "
        "credencial vencida, casco no disponible). Sin humano que resuelva, el kiosco queda "
        "<b>completamente bloqueado de 3 a 5 minutos</b>. Durante ese tiempo ese kiosco no "
        "atiende a nadie mas. Si coinciden dos bloqueos simultaneos, la capacidad cae a "
        "1 kiosco y la cola crece rapidamente.", st_bod))

    # ── SEC 2: Datos de llegadas detallados ──────────────────────────────────
    story.append(Paragraph("2. Llegadas Registradas — Trenes y Lotes de Alumnos", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    story.append(Paragraph(
        "La siguiente tabla muestra exactamente cuantos alumnos llego cada tren durante "
        "la hora pico. Cada fila es un evento real de llegada masiva al modulo.", st_bod))

    # Tabla de trenes con datos reales del campo
    horas_base = 5*60 + 45  # 05:45 en minutos
    tren_hdr = ["Tren #", "Hora aprox.", "Intervalo desde anterior", "Alumnos en el lote", "Acumulado llegados"]
    tren_rows = [tren_hdr]
    acum = 0
    tiempos_acum = 0
    for i, (inter, lote) in enumerate(zip(INTER_LLEGADAS_MIN, PERSONAS_POR_TREN)):
        tiempos_acum += inter
        acum += lote
        hora_min = horas_base + tiempos_acum
        hora_str = f"{int(hora_min//60):02d}:{int(hora_min%60):02d}"
        tren_rows.append([
            f"T{i+1:02d}", hora_str,
            f"{inter} min",
            f"{lote} alumno{'s' if lote > 1 else ''}",
            f"{acum} acumulados"
        ])
    story.append(tabla(tren_rows, [0.7*inch, 1.1*inch, 1.8*inch, 1.8*inch, 1.9*inch],
                       hdr_color=verde, row_colors=[colors.white, verd_c]))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        f"Total de trenes registrados: {len(INTER_LLEGADAS_MIN)} | "
        f"Total de alumnos en los lotes: {sum(PERSONAS_POR_TREN)} | "
        f"Intervalo promedio entre trenes: {np.mean(INTER_LLEGADAS_MIN):.1f} min | "
        f"Lote promedio: {np.mean(PERSONAS_POR_TREN):.1f} alumnos/tren", st_not))

    story.append(Paragraph("2.1 Parametros del modelo de llegadas", st_h2))
    llegadas_params = [
        ["Parametro", "Valor", "Significado"],
        ["Distribucion inter-arribo",   "Exponencial",         "Tiempo entre trenes sigue proceso de Poisson"],
        ["Media inter-arribo (campo)",  f"{np.mean(INTER_LLEGADAS_MIN):.1f} min", "Promedio real medido en sitio"],
        ["Tasa de trenes (lambda)",     f"{LAMBDA_TRENES_S*60:.4f} trenes/min", "Cuantos trenes llegan por minuto en promedio"],
        ["Tamano de lote",              "Uniforme(1, 8)",       "Cada tren trae entre 1 y 8 alumnos al modulo"],
        ["Media del lote",              f"{MEDIA_LOTE:.1f} alumnos", "Promedio de personas por ráfaga"],
        ["Tasa efectiva (lambda_eff)",  f"{LAMBDA_EFF_S*60:.3f} alumnos/min", "Llegadas reales al modulo por minuto"],
        ["Paciencia del alumno",        "Exponencial (media 5 min)", "Tiempo maximo que espera antes de irse"],
    ]
    story.append(tabla(llegadas_params, [2.1*inch, 1.8*inch, 2.8*inch]))
    story.append(PageBreak())

    # ── SEC 3: Configuracion de kioscos ──────────────────────────────────────
    story.append(Paragraph("3. Configuracion de los Kioscos", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    kiosc_data = [
        ["Parametro", "Valor", "Detalle"],
        ["Numero de kioscos",      "3 en paralelo",             "Atienden simultaneamente (simpy.Resource capacity=3)"],
        ["Servicio normal",        "Uniforme(10, 20) s",        "Lectura RFID + verificacion NIP + asignacion bici"],
        ["Servicio promedio",      "15 s por alumno",           "4x mas rapido que el servidor manual (51 s)"],
        ["Probabilidad excepcion", "5% por alumno",             "NIP bloqueado, credencial vencida, sin casco"],
        ["Duracion del bloqueo",   "Uniforme(3, 5) min",        "El kiosco queda fuera de servicio sin soporte humano"],
        ["Resolucion de excepcion","No disponible",             "Sin humano, el alumno simplemente abandona el sistema"],
        ["Distribucion llegadas",  "M[X] Poisson compuesto",   "Misma que Escenario 1 — misma realidad de campo"],
    ]
    story.append(tabla(kiosc_data, [1.9*inch, 1.7*inch, 3.1*inch],
                       hdr_color=naranj, row_colors=[colors.white, narj_c]))

    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        "<b>Notacion de Kendall:</b> M[X] / U(10,20) / 3 / inf + BLOQUEO(p=0.05, t=3-5 min) + RENEGING", st_not))
    story.append(Paragraph(
        "Donde M[X] = llegadas en rafaga Poisson, U(10,20) = servicio uniforme, "
        "3 = tres servidores, BLOQUEO = kiosco fuera de servicio por excepcion, "
        "RENEGING = abandono por impaciencia.", st_not))

    # ── SEC 4: Resultados ────────────────────────────────────────────────────
    story.append(Paragraph("4. Resultados de la Simulacion", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    story.append(Paragraph("4.1 Flujo de alumnos — que paso con cada persona", st_h2))
    story.append(Paragraph(
        "De todos los alumnos generados por los trenes durante las 2 horas, "
        "la siguiente tabla explica exactamente que le paso a cada grupo:", st_bod))

    tb_bloq = m.tiempo_bloqueado()
    flujo_data = [
        ["Grupo de alumnos", "Cantidad", "Por que", "Consecuencia"],
        ["Llegaron al modulo",
         str(m.total),
         "Generados por los trenes durante 2 horas de hora pico",
         "Punto de partida — 100% de la demanda"],
        ["Atendidos exitosamente",
         f"{m.atendidos}  ({100-m.p_abandono*100:.1f}%)",
         "Esperaron su turno y el kiosco los atendio sin excepcion",
         "Obtuvieron su bicicleta"],
        ["Abandonaron por impaciencia",
         f"{sum(1 for a in m.alumnos if a.abandono and not a.fue_excepcion)}",
         "Esperaron mas de su limite de paciencia (~5 min) en la cola",
         "Se fueron sin bicicleta — llegaron tarde a clase"],
        ["Abandonaron por excepcion",
         f"{m.excepciones}",
         "5% de probabilidad: NIP bloqueado, credencial vencida o sin casco",
         "El kiosco los expulso Y se bloqueo 3-5 min"],
        ["Total abandonos",
         f"{m.abandonos}  ({m.p_abandono:.1%})",
         "Suma de abandonos por impaciencia + por excepcion",
         "Tasa de abandono total del sistema"],
    ]
    t_flujo = Table(flujo_data, colWidths=[1.6*inch, 1.2*inch, 2.3*inch, 1.6*inch])
    t_flujo.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), azul),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, azul_c, colors.white,
                                            narj_c, colors.HexColor("#FFEBEE")]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ('VALIGN',        (0,0), (-1,-1), 'TOP'),
        ('FONTNAME',      (0,5), (-1,5), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (1,5), (1,5), rojo),
    ]))
    story.append(t_flujo)
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("4.2 Metricas de rendimiento del sistema", st_h2))
    metr_data = [
        ["Metrica", "Valor", "Significado practico"],
        ["Throughput",             f"{m.throughput(SIM_TIME_S):.1f} bici/hora",
         "Bicicletas entregadas por hora — capacidad real del sistema"],
        ["Wq — Espera promedio",   f"{m.wq_prom:.1f} s  ({m.wq_prom/60:.1f} min)",
         "Tiempo promedio que un alumno espera en la cola antes de ser atendido"],
        ["Wq — Espera maxima",     f"{m.wq_max:.1f} s  ({m.wq_max/60:.1f} min)",
         "El peor tiempo de espera registrado en toda la simulacion"],
        ["W — Tiempo en sistema",  f"{m.w_prom:.1f} s",
         "Espera en cola + tiempo de servicio en el kiosco"],
        ["Lq — Cola promedio",     f"{m.lq_prom:.2f} alumnos",
         "Promedio de personas esperando en cualquier momento"],
        ["Lq — Cola maxima",       f"{m.lq_max} alumnos",
         "Pico maximo de la fila durante las 2 horas"],
        ["Bloqueos totales",       f"{len(m.bloqueos)} eventos",
         "Veces que un kiosco quedo fuera de servicio por excepcion"],
        ["Tiempo total bloqueado", f"{tb_bloq:.0f} s  ({tb_bloq/60:.1f} min)",
         f"De 120 min de hora pico, {tb_bloq/60:.1f} min se perdieron por bloqueos"],
        ["Error Ley de Little",    f"{m.little_error():.1f}%",
         "< 15% confirma que la simulacion es matematicamente valida"],
    ]
    t_metr = Table(metr_data, colWidths=[1.9*inch, 1.7*inch, 3.1*inch])
    t_metr.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), azul),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, azul_c]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('TEXTCOLOR',     (1,8), (1,8), naranj),
        ('FONTNAME',      (1,8), (1,8), 'Helvetica-Bold'),
    ]))
    story.append(t_metr)
    story.append(Spacer(1, 0.15*inch))

    # Grafica cola
    story.append(Image(rutas['cola'], width=6.5*inch, height=2.6*inch, hAlign='CENTER'))
    story.append(Spacer(1, 0.1*inch))
    story.append(Image(rutas['kioscos'], width=6.5*inch, height=2.2*inch, hAlign='CENTER'))
    story.append(PageBreak())

    # ── SEC 5: Detalle de bloqueos ───────────────────────────────────────────
    story.append(Paragraph("5. Registro Detallado de Bloqueos", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=naranj, spaceAfter=8))
    story.append(Paragraph(
        "Cada vez que un alumno tuvo una excepcion, el kiosco quedo bloqueado. "
        "Aqui se muestra cada evento de bloqueo registrado durante la simulacion, "
        "con su duracion exacta y el impacto en la hora del dia:", st_bod))

    if m.bloqueos:
        blq_hdr = ["Bloqueo #", "Inicio (min)", "Fin (min)", "Duracion", "Hora aprox.", "Impacto"]
        blq_rows = [blq_hdr]
        for i, b in enumerate(m.bloqueos):
            ini_min  = b["inicio"] / 60
            fin_min  = b["fin"]    / 60
            dur_min  = b["dur"]    / 60
            hora_ini = horas_base + ini_min
            hora_str = f"{int(hora_ini//60):02d}:{int(hora_ini%60):02d}"
            # Cuantos alumnos llegaron durante el bloqueo
            alumnos_durante = sum(
                1 for a in m.alumnos
                if b["inicio"] <= a.t_llegada <= b["fin"]
            )
            blq_rows.append([
                f"#{i+1}",
                f"{ini_min:.1f} min",
                f"{fin_min:.1f} min",
                f"{dur_min:.1f} min ({b['dur']:.0f} s)",
                hora_str,
                f"~{alumnos_durante} alumnos llegaron mientras el kiosco estaba caido"
            ])
        t_blq = Table(blq_rows, colWidths=[0.7*inch, 1.1*inch, 0.9*inch, 1.3*inch, 0.9*inch, 2.4*inch])
        t_blq.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), naranj),
            ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
            ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE',      (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, narj_c]),
            ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
            ('TOPPADDING',    (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ]))
        story.append(t_blq)
    else:
        story.append(Paragraph("No se registraron bloqueos en esta corrida.", st_not))

    story.append(Spacer(1, 0.15*inch))
    story.append(Image(rutas['esperas'], width=5.5*inch, height=2.8*inch, hAlign='CENTER'))

    # ── SEC 6: Replicas ──────────────────────────────────────────────────────
    story.append(Paragraph("6. Estabilidad Estadistica — 10 Replicas", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    story.append(Paragraph(
        "Para confirmar que los resultados son consistentes y no dependen de "
        "una semilla aleatoria particular, se corrieron 10 simulaciones independientes:", st_bod))

    ic = lambda arr: 1.96 * np.std(arr) / np.sqrt(len(arr))
    rep_tabla = [
        ["Metrica", "Media", "Desv. Std.", "IC 95%", "Interpretacion"],
        ["P_abandono",
         f"{np.mean(rep_stats['abandonos']):.1f}%",
         f"±{np.std(rep_stats['abandonos']):.1f}%",
         f"[{np.mean(rep_stats['abandonos'])-ic(rep_stats['abandonos']):.1f}%, "
         f"{np.mean(rep_stats['abandonos'])+ic(rep_stats['abandonos']):.1f}%]",
         "Porcentaje de alumnos que no obtienen bicicleta"],
        ["Wq promedio",
         f"{np.mean(rep_stats['wq']):.1f} s",
         f"±{np.std(rep_stats['wq']):.1f} s",
         f"[{np.mean(rep_stats['wq'])-ic(rep_stats['wq']):.0f}, "
         f"{np.mean(rep_stats['wq'])+ic(rep_stats['wq']):.0f}] s",
         "Tiempo de espera consistente entre replicas"],
        ["Throughput",
         f"{np.mean(rep_stats['throughput']):.1f} bici/h",
         f"±{np.std(rep_stats['throughput']):.1f}",
         f"[{np.mean(rep_stats['throughput'])-ic(rep_stats['throughput']):.1f}, "
         f"{np.mean(rep_stats['throughput'])+ic(rep_stats['throughput']):.1f}]",
         "Bicicletas entregadas por hora"],
        ["Bloqueos/turno",
         f"{np.mean(rep_stats['bloqueos']):.1f}",
         f"±{np.std(rep_stats['bloqueos']):.1f}",
         f"[{np.mean(rep_stats['bloqueos'])-ic(rep_stats['bloqueos']):.1f}, "
         f"{np.mean(rep_stats['bloqueos'])+ic(rep_stats['bloqueos']):.1f}]",
         "Eventos de bloqueo por cada 2 horas de operacion"],
    ]
    story.append(tabla(rep_tabla, [1.3*inch, 1.1*inch, 1.0*inch, 1.6*inch, 1.7*inch]))
    story.append(Spacer(1, 0.1*inch))
    story.append(Image(rutas['replicas'], width=6.5*inch, height=2.6*inch, hAlign='CENTER'))

    # ── SEC 7: Conclusiones ──────────────────────────────────────────────────
    story.append(Paragraph("7. Conclusiones del Escenario 2", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))

    conclusiones = [
        ("Mejora real vs Sistema manual",
         f"La tasa de abandono baja de ~35% (Escenario 1) a "
         f"{np.mean(rep_stats['abandonos']):.1f}% gracias a los 3 kioscos en paralelo "
         f"y el servicio 4 veces mas rapido (15 s vs 51 s)."),
        ("El bloqueo es el talon de Aquiles",
         f"Se registran en promedio {np.mean(rep_stats['bloqueos']):.1f} bloqueos por "
         f"turno de 2 horas, con una duracion de 3 a 5 min cada uno. Cuando coinciden "
         f"dos bloqueos la capacidad cae a 1 kiosco y la cola explota."),
        ("Tiempo productivo perdido",
         f"De 120 minutos de hora pico, {m.tiempo_bloqueado()/60:.1f} minutos se pierden "
         f"en bloqueos (en la corrida principal). Eso es tiempo en que los alumnos en cola "
         f"simplemente esperan sin avanzar."),
        ("El 5% genera el 100% de los problemas graves",
         "Los bloqueos son causados por una minoria de casos, pero su impacto se "
         "multiplica porque afectan a todos los que estan esperando en ese momento."),
        ("Necesidad del Escenario 3",
         "La solucion no es eliminar los kioscos sino agregar un humano que resuelva "
         "las excepciones SIN bloquear el kiosco. El Escenario 3 demuestra exactamente eso."),
    ]
    for titulo_c, texto_c in conclusiones:
        story.append(Paragraph(f"<b>{titulo_c}:</b> {texto_c}", st_bod))

    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC")))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        f"Reporte generado automaticamente por scen2_autonomo.py · "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')} · "
        "UNAM Ingenieria en Telecomunicaciones", st_pie))

    doc.build(story)
    return ruta_pdf

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLA LIMPIA
# ══════════════════════════════════════════════════════════════════════════════
def imprimir_resultados(m, rep_stats):
    sep = "─" * 52
    print("\n" + "═"*52)
    print("  PUMABICIS · Escenario 2 · Sistema Autonomo")
    print("═"*52)
    print("\n  FLUJO DE ALUMNOS")
    print(sep)
    print(f"  Llegaron al modulo  : {m.total} alumnos")
    print(f"  Fueron atendidos    : {m.atendidos} alumnos")
    print(f"  Abandonaron         : {m.abandonos} alumnos  ({m.p_abandono:.1%})")
    print(f"    - Por impaciencia : {sum(1 for a in m.alumnos if a.abandono and not a.fue_excepcion)}")
    print(f"    - Por excepcion   : {m.excepciones}  (bloquearon el kiosco)")
    print(f"  Bicicletas/hora     : {m.throughput(SIM_TIME_S):.1f}")
    print("\n  BLOQUEOS DE KIOSCO")
    print(sep)
    print(f"  Bloqueos totales    : {len(m.bloqueos)} eventos")
    print(f"  Tiempo bloqueado    : {m.tiempo_bloqueado()/60:.1f} min de 120 min totales")
    print("\n  TIEMPOS DE ESPERA")
    print(sep)
    print(f"  Espera promedio     : {m.wq_prom:.0f} s  ({m.wq_prom/60:.1f} min)")
    print(f"  Espera maxima       : {m.wq_max:.0f} s  ({m.wq_max/60:.1f} min)")
    print(f"  Tiempo en sistema   : {m.w_prom:.0f} s")
    print("\n  COLA")
    print(sep)
    print(f"  Cola promedio (Lq)  : {m.lq_prom:.1f} alumnos")
    print(f"  Cola maxima         : {m.lq_max} alumnos")
    print("\n  VALIDACION")
    print(sep)
    err = m.little_error()
    print(f"  Error Ley de Little : {err:.1f}%  {'✓ Valido' if err < 15 else '⚠ Revisar'}")
    print("\n  ESTABILIDAD (10 replicas)")
    print(sep)
    ic = lambda arr: 1.96 * np.std(arr) / np.sqrt(len(arr))
    print(f"  Abandono promedio   : {np.mean(rep_stats['abandonos']):.1f}%  IC95: +-{ic(rep_stats['abandonos']):.1f}%")
    print(f"  Bloqueos promedio   : {np.mean(rep_stats['bloqueos']):.1f} por turno  IC95: +-{ic(rep_stats['bloqueos']):.1f}")
    print(f"  Throughput promedio : {np.mean(rep_stats['throughput']):.1f} bici/h  IC95: +-{ic(rep_stats['throughput']):.1f}")
    print()

# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n[1/4] Corriendo simulacion principal...")
    m = correr_simulacion(RANDOM_SEED)

    print("[2/4] Corriendo 10 replicas...")
    rep_stats = {'abandonos': [], 'wq': [], 'throughput': [], 'bloqueos': []}
    for seed in range(N_REPLICAS):
        mr = correr_simulacion(seed)
        rep_stats['abandonos'].append(mr.p_abandono * 100)
        rep_stats['wq'].append(mr.wq_prom)
        rep_stats['throughput'].append(mr.throughput(SIM_TIME_S))
        rep_stats['bloqueos'].append(len(mr.bloqueos))

    print("[3/4] Generando graficas...")
    rutas = generar_graficas(m, rep_stats)

    print("[4/4] Generando reporte PDF...")
    ruta_pdf = generar_pdf(m, rep_stats, rutas)

    imprimir_resultados(m, rep_stats)
    print(f"  Reporte PDF  -> {ruta_pdf}")
    print(f"  Carpeta      -> {CARPETA}/")
    print("="*52)
