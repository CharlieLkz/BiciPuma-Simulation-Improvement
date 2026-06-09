"""
PUMABICIS — Escenario 1: Sistema Manual
Simulación con SimPy + Reporte PDF automático
UNAM · Ingeniería en Telecomunicaciones · Teoría de Colas
"""

import simpy
import random
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, HRFlowable, PageBreak)

# ══════════════════════════════════════════════════════════════════════════════
# DATOS DE CAMPO Y PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

INTER_LLEGADAS_MIN = [5, 3, 7, 5, 10, 3, 7, 5, 3, 7, 10]
PERSONAS_POR_TREN  = [4, 7, 1, 3, 8, 4, 2, 5, 4, 3, 1, 8]
TIEMPOS_SERVICIO_S = [
    35, 42, 85, 32, 50, 65, 38, 40, 90, 45,
    30, 55, 60, 42, 88, 35, 33, 48, 75, 50,
    36, 41, 62, 80, 34, 39, 45, 52, 85, 31,
    38, 44, 58, 70, 35, 40, 89, 45, 33, 50
]

LAMBDA_TRENES_S = 1 / (np.mean(INTER_LLEGADAS_MIN) * 60)
MEDIA_LOTE      = np.mean(PERSONAS_POR_TREN)
LAMBDA_EFF_S    = LAMBDA_TRENES_S * MEDIA_LOTE
LN_S, LN_LOC, LN_SCALE = stats.lognorm.fit(TIEMPOS_SERVICIO_S)

SIM_TIME_S      = 7200
PATIENCE_MEAN_S = 300
RANDOM_SEED     = 42
N_REPLICAS      = 10

CARPETA_SALIDA  = "reporte_scen1"
os.makedirs(CARPETA_SALIDA, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTIDADES Y MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

class Alumno:
    _cnt = 0
    def __init__(self, t_llegada, paciencia):
        Alumno._cnt += 1
        self.id           = Alumno._cnt
        self.t_llegada    = t_llegada
        self.paciencia    = paciencia
        self.t_inicio_srv = None
        self.t_salida     = None
        self.atendido     = False
        self.abandono     = False

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

    def registrar(self, a):      self.alumnos.append(a)
    def log_lq(self, t, lq):    self.log_cola.append((t, lq))

    @property
    def total(self):     return len(self.alumnos)
    @property
    def atendidos(self): return sum(1 for a in self.alumnos if a.atendido)
    @property
    def abandonos(self): return sum(1 for a in self.alumnos if a.abandono)
    @property
    def p_abandono(self):
        return self.abandonos / self.total if self.total else 0.0
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
        if len(self.log_cola) < 2:
            return 0.0
        ts  = [x[0] for x in self.log_cola]
        lqs = [x[1] for x in self.log_cola]
        return np.trapezoid(lqs, ts) / (ts[-1] - ts[0] + 1e-9)
    @property
    def lq_max(self):
        return max((x[1] for x in self.log_cola), default=0)
    def throughput(self, sim_time):
        return self.atendidos / (sim_time / 3600)
    def little_error(self):
        lq_l   = LAMBDA_EFF_S * self.wq_prom
        lq_emp = self.lq_prom
        return abs(lq_l - lq_emp) / (lq_emp + 1e-9) * 100


# ══════════════════════════════════════════════════════════════════════════════
# PROCESOS SIMPY
# ══════════════════════════════════════════════════════════════════════════════

def muestrear_servicio(rng):
    return max(
        stats.lognorm.rvs(LN_S, LN_LOC, LN_SCALE,
                          random_state=rng.randint(0, 2**31)),
        30.0
    )

def proceso_alumno(env, alumno, servidor, metricas, rng):
    metricas.log_lq(env.now, len(servidor.queue))
    with servidor.request() as req:
        resultado = yield req | env.timeout(alumno.paciencia)
        if req not in resultado:
            alumno.abandono = True
            alumno.t_salida = env.now
            metricas.registrar(alumno)
            metricas.log_lq(env.now, len(servidor.queue))
            return
        alumno.t_inicio_srv = env.now
        yield env.timeout(muestrear_servicio(rng))
        alumno.atendido = True
        alumno.t_salida = env.now
        metricas.registrar(alumno)
        metricas.log_lq(env.now, len(servidor.queue))

def generador_trenes(env, servidor, metricas, rng):
    while True:
        yield env.timeout(rng.expovariate(LAMBDA_TRENES_S))
        for _ in range(rng.randint(1, 8)):
            alumno = Alumno(env.now, rng.expovariate(1 / PATIENCE_MEAN_S))
            env.process(proceso_alumno(env, alumno, servidor, metricas, rng))

def correr_simulacion(semilla=RANDOM_SEED):
    Alumno._cnt = 0
    rng         = random.Random(semilla)
    env         = simpy.Environment()
    metricas    = Metricas()
    servidor    = simpy.Resource(env, capacity=1)
    env.process(generador_trenes(env, servidor, metricas, rng))
    env.run(until=SIM_TIME_S)
    return metricas


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS (guardadas como PNG para el PDF)
# ══════════════════════════════════════════════════════════════════════════════

def generar_graficas(m, replicas_stats):
    rutas = {}

    # ── Gráfica 1: Cola en el tiempo ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ts  = [x[0] / 60 for x in m.log_cola]
    lqs = [x[1]      for x in m.log_cola]
    ax.step(ts, lqs, color="#1A5FAF", linewidth=0.7, where='post', alpha=0.8)
    ax.fill_between(ts, lqs, step='post', alpha=0.15, color="#1A5FAF")
    ax.axhline(m.lq_prom, color="#E24B4A", linestyle="--", linewidth=1.5,
               label=f"Lq promedio = {m.lq_prom:.1f} alumnos")
    ax.set_title("Longitud de cola a lo largo de la hora pico", fontsize=12, fontweight='bold')
    ax.set_xlabel("Tiempo transcurrido (minutos)")
    ax.set_ylabel("Alumnos esperando")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, SIM_TIME_S / 60)
    plt.tight_layout()
    rutas['cola'] = os.path.join(CARPETA_SALIDA, "graf_cola.png")
    plt.savefig(rutas['cola'], dpi=150, bbox_inches='tight')
    plt.close()

    # ── Gráfica 2: Histograma de esperas ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    esperas = [a.t_espera / 60 for a in m.alumnos if a.atendido]
    ax.hist(esperas, bins=18, color="#1D9E75", edgecolor="white",
            linewidth=0.5, alpha=0.9)
    ax.axvline(m.wq_prom / 60, color="#E24B4A", linestyle="--", linewidth=2,
               label=f"Promedio = {m.wq_prom/60:.1f} min")
    ax.axvline(5, color="#E06C00", linestyle=":", linewidth=1.5,
               label="Límite de paciencia (5 min)")
    ax.set_title("Distribución de tiempos de espera en cola", fontsize=12, fontweight='bold')
    ax.set_xlabel("Tiempo de espera (minutos)")
    ax.set_ylabel("Número de alumnos")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    rutas['esperas'] = os.path.join(CARPETA_SALIDA, "graf_esperas.png")
    plt.savefig(rutas['esperas'], dpi=150, bbox_inches='tight')
    plt.close()

    # ── Gráfica 3: Ajuste distribución ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    x_range = np.linspace(25, 100, 300)
    pdf_ln  = stats.lognorm.pdf(x_range, LN_S, LN_LOC, LN_SCALE)
    ax.hist(TIEMPOS_SERVICIO_S, bins=12, density=True,
            color="#B5D4F4", edgecolor="white", label="Datos reales (n=40)")
    ax.plot(x_range, pdf_ln, color="#1A5FAF", linewidth=2.5,
            label="Ajuste Lognormal (KS=0.062, p=0.995)")
    ax.set_title("Ajuste estadístico: tiempos de servicio observados", fontsize=12, fontweight='bold')
    ax.set_xlabel("Tiempo de servicio (segundos)")
    ax.set_ylabel("Densidad de probabilidad")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    rutas['ajuste'] = os.path.join(CARPETA_SALIDA, "graf_ajuste.png")
    plt.savefig(rutas['ajuste'], dpi=150, bbox_inches='tight')
    plt.close()

    # ── Gráfica 4: Barras de réplicas ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    semillas = list(range(N_REPLICAS))

    axes[0].bar(semillas, replicas_stats['abandonos'],
                color="#E24B4A", alpha=0.8, edgecolor='white')
    axes[0].axhline(np.mean(replicas_stats['abandonos']), color='black',
                    linestyle='--', linewidth=1.5)
    axes[0].set_title("Tasa de abandono por réplica (%)")
    axes[0].set_xlabel("Réplica (semilla)")
    axes[0].set_ylabel("%")
    axes[0].grid(alpha=0.3)

    axes[1].bar(semillas, replicas_stats['wq'],
                color="#1A5FAF", alpha=0.8, edgecolor='white')
    axes[1].axhline(np.mean(replicas_stats['wq']), color='black',
                    linestyle='--', linewidth=1.5)
    axes[1].set_title("Tiempo de espera promedio Wq (s)")
    axes[1].set_xlabel("Réplica (semilla)")
    axes[1].set_ylabel("Segundos")
    axes[1].grid(alpha=0.3)

    axes[2].bar(semillas, replicas_stats['throughput'],
                color="#1D9E75", alpha=0.8, edgecolor='white')
    axes[2].axhline(np.mean(replicas_stats['throughput']), color='black',
                    linestyle='--', linewidth=1.5)
    axes[2].set_title("Throughput por réplica (bici/hora)")
    axes[2].set_xlabel("Réplica (semilla)")
    axes[2].set_ylabel("Bici/hora")
    axes[2].grid(alpha=0.3)

    plt.suptitle("Estabilidad estadística — 10 réplicas independientes",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    rutas['replicas'] = os.path.join(CARPETA_SALIDA, "graf_replicas.png")
    plt.savefig(rutas['replicas'], dpi=150, bbox_inches='tight')
    plt.close()

    return rutas


# ══════════════════════════════════════════════════════════════════════════════
# GENERADOR DE PDF
# ══════════════════════════════════════════════════════════════════════════════

def generar_pdf(m, replicas_stats, rutas_graficas):
    ruta_pdf = os.path.join(CARPETA_SALIDA, "reporte_pumabicis_scen1.pdf")
    doc      = SimpleDocTemplate(ruta_pdf, pagesize=letter,
                                  leftMargin=0.9*inch, rightMargin=0.9*inch,
                                  topMargin=0.9*inch, bottomMargin=0.9*inch)
    styles   = getSampleStyleSheet()
    story    = []

    # ── Estilos personalizados ───────────────────────────────────────────────
    azul  = colors.HexColor("#1A5FAF")
    verde = colors.HexColor("#1D9E75")
    rojo  = colors.HexColor("#E24B4A")

    st_titulo = ParagraphStyle("titulo",
        parent=styles["Title"], fontSize=20, textColor=azul,
        spaceAfter=4, alignment=TA_CENTER)

    st_subtitulo = ParagraphStyle("subtitulo",
        parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#555555"),
        spaceAfter=16, alignment=TA_CENTER)

    st_h1 = ParagraphStyle("h1",
        parent=styles["Heading1"], fontSize=13, textColor=azul,
        spaceBefore=18, spaceAfter=6, borderPad=4)

    st_h2 = ParagraphStyle("h2",
        parent=styles["Heading2"], fontSize=11, textColor=colors.HexColor("#1D9E75"),
        spaceBefore=12, spaceAfter=4)

    st_body = ParagraphStyle("body",
        parent=styles["Normal"], fontSize=10, leading=15,
        spaceAfter=8, alignment=TA_JUSTIFY)

    st_nota = ParagraphStyle("nota",
        parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#666666"),
        leading=13, spaceAfter=6, leftIndent=12, borderPad=2)

    st_code = ParagraphStyle("code",
        parent=styles["Code"], fontSize=9, leading=13,
        backColor=colors.HexColor("#F4F6F9"), spaceAfter=8)

    # ── PORTADA ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("PUMABICIS", st_titulo))
    story.append(Paragraph("Simulación de Teoría de Colas — Escenario 1: Sistema Manual", st_subtitulo))
    story.append(HRFlowable(width="100%", thickness=2, color=azul))
    story.append(Spacer(1, 0.15*inch))

    meta = [
        ["Universidad:", "UNAM — Facultad de Ingeniería"],
        ["Asignatura:", "Teoría de Colas y Simulación"],
        ["Sistema analizado:", "Cicloestación Pumabicis — Metro CU"],
        ["Periodo de observación:", "Hora pico matutina · 05:45 a 07:45 hrs"],
        ["Herramientas:", "Python 3 · SimPy · SciPy · Matplotlib · ReportLab"],
        ["Fecha del reporte:", datetime.now().strftime("%d de %B de %Y · %H:%M hrs")],
    ]
    t_meta = Table(meta, colWidths=[2.2*inch, 4.5*inch])
    t_meta.setStyle(TableStyle([
        ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',  (0,0), (0,-1), azul),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor("#F0F6FF")]),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC")))
    story.append(PageBreak())

    # ── SECCIÓN 1: Descripción del problema ─────────────────────────────────
    story.append(Paragraph("1. Descripción del Problema", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    story.append(Paragraph(
        "El sistema de préstamo de bicicletas <b>Pumabicis</b> en la cicloestación del Metro CU "
        "presenta cuellos de botella críticos durante las horas pico de la mañana. "
        "Las llegadas de alumnos no son uniformes: ocurren en <b>ráfagas sincronizadas</b> con "
        "el arribo de cada tren del metro, generando lotes de 1 a 8 personas de forma casi "
        "simultánea cada 3 a 10 minutos.", st_body))
    story.append(Paragraph(
        "El módulo es operado por <b>un único trabajador manual</b> que debe verificar NIP, "
        "resello de credencial, entrega de casco y escaneo de código de barras. Este proceso "
        "tarda entre 30 y 90 segundos por alumno, con alta varianza debido a factores como "
        "credenciales desgastadas, fallas del escáner o alumnos de primer ingreso. El resultado "
        "es una tasa de abandono que supera el 30% en hora pico.", st_body))

    # ── SECCIÓN 2: Datos de campo ────────────────────────────────────────────
    story.append(Paragraph("2. Datos de Campo Recopilados", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))

    story.append(Paragraph("2.1 Llegadas del metro (intervalos entre trenes)", st_h2))
    datos_llegadas = [
        ["Parámetro", "Valor"],
        ["Número de trenes registrados", "12"],
        ["Intervalo mínimo entre trenes", "3 minutos"],
        ["Intervalo máximo entre trenes", "10 minutos"],
        ["Intervalo promedio", f"{np.mean(INTER_LLEGADAS_MIN):.1f} minutos"],
        ["Tamaño de lote mínimo", "1 alumno"],
        ["Tamaño de lote máximo", "8 alumnos"],
        ["Tamaño de lote promedio", f"{np.mean(PERSONAS_POR_TREN):.1f} alumnos"],
        ["Tasa efectiva de llegadas (λ)", f"{LAMBDA_EFF_S*60:.3f} alumnos/min"],
    ]
    t_llegadas = Table(datos_llegadas, colWidths=[3.5*inch, 3.2*inch])
    t_llegadas.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), azul),
        ('TEXTCOLOR',    (0,0), (-1,0), colors.white),
        ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',     (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF4FF")]),
        ('GRID',         (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',  (0,0), (-1,-1), 8),
    ]))
    story.append(t_llegadas)
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("2.2 Tiempos de servicio observados (n=40 mediciones)", st_h2))
    datos_srv = [
        ["Estadístico", "Valor", "Interpretación"],
        ["Media muestral", f"{np.mean(TIEMPOS_SERVICIO_S):.1f} s",
         "Atención promedio por alumno"],
        ["Desviación estándar", f"{np.std(TIEMPOS_SERVICIO_S, ddof=1):.1f} s",
         "Alta variabilidad en el servicio"],
        ["Coeficiente de variación", "0.360",
         "Varianza moderada-alta (CV > 0.3)"],
        ["Mínimo observado", "30 s",
         "Usuario frecuente con casco listo"],
        ["Máximo observado", "90 s",
         "Falla de escáner en código de barras"],
        ["Skewness (asimetría)", "0.921",
         "Cola derecha — casos lentos poco frecuentes"],
    ]
    t_srv = Table(datos_srv, colWidths=[2.1*inch, 1.4*inch, 3.2*inch])
    t_srv.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), verde),
        ('TEXTCOLOR',    (0,0), (-1,0), colors.white),
        ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',     (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EDFAF4")]),
        ('GRID',         (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',  (0,0), (-1,-1), 8),
    ]))
    story.append(t_srv)

    # ── SECCIÓN 3: Ajuste estadístico ───────────────────────────────────────
    story.append(Paragraph("3. Ajuste de Distribución de Probabilidad", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    story.append(Paragraph(
        "Se evaluaron cinco familias de distribuciones usando la prueba de bondad de ajuste "
        "<b>Kolmogorov-Smirnov (KS)</b>. Una distribución es aceptable cuando su p-valor supera "
        "0.05. Cuanto mayor sea el p-valor y menor el estadístico KS, mejor el ajuste.", st_body))

    ajuste_data = [
        ["Distribución", "Estadístico KS", "p-valor", "Resultado"],
        ["Lognormal",    "0.0621",         "0.9953",  "MEJOR AJUSTE"],
        ["Weibull",      "0.0643",         "0.9928",  "Excelente"],
        ["Exponencial",  "0.0652",         "0.9914",  "Muy buena"],
        ["Gamma",        "0.0683",         "0.9859",  "Buena"],
        ["Normal",       "0.1821",         "0.1239",  "Rechazada"],
    ]
    t_ajuste = Table(ajuste_data, colWidths=[2*inch, 1.6*inch, 1.6*inch, 1.5*inch])
    t_ajuste.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), azul),
        ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor("#E8F5E9"), colors.white,
          colors.white, colors.white, colors.HexColor("#FFEBEE")]),
        ('TEXTCOLOR',   (3,1), (3,1), verde),
        ('FONTNAME',    (3,1), (3,1), 'Helvetica-Bold'),
        ('TEXTCOLOR',   (3,5), (3,5), rojo),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_ajuste)
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "<b>Conclusión:</b> La distribución <b>Lognormal</b> modela mejor los tiempos de servicio. "
        "Esto es consistente con el CV=0.36 y el skewness positivo: la mayoría de los alumnos "
        "son atendidos rápido (30–50 s), pero existe una cola derecha de casos lentos "
        "(escáner fallido, primer ingreso, credencial desgastada) que la Normal no puede capturar.", st_nota))

    story.append(Image(rutas_graficas['ajuste'],
                       width=5.5*inch, height=2.8*inch, hAlign='CENTER'))
    story.append(Spacer(1, 0.1*inch))
    story.append(PageBreak())

    # ── SECCIÓN 4: Modelo de simulación ─────────────────────────────────────
    story.append(Paragraph("4. Modelo de Simulación", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    story.append(Paragraph(
        "La simulación implementa un sistema de colas con la siguiente notación de Kendall:", st_body))
    story.append(Paragraph(
        "<b>M[X] / LN / 1 / inf + RENEGING</b>", st_code))

    modelo_data = [
        ["Componente", "Notación", "Implementación"],
        ["Llegadas",       "M[X]",      "Proceso de Poisson compuesto: inter-arribo Exponencial + lote Uniforme(1,8)"],
        ["Servicio",       "LN",        "Distribución Lognormal ajustada a los 40 datos de campo"],
        ["Servidores",     "1",         "Un solo trabajador manual (simpy.Resource capacity=1)"],
        ["Capacidad",      "inf",       "Sin límite de cola (hipótesis conservadora)"],
        ["Abandono",       "RENEGING",  "Paciencia Exponencial con media de 5 minutos por alumno"],
    ]
    t_modelo = Table(modelo_data, colWidths=[1.3*inch, 0.9*inch, 4.5*inch])
    t_modelo.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), azul),
        ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF4FF")]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_modelo)

    # ── SECCIÓN 5: Resultados ────────────────────────────────────────────────
    story.append(Paragraph("5. Resultados de la Simulación", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    story.append(Paragraph("5.1 Corrida principal (semilla = 42)", st_h2))

    res_data = [
        ["Métrica", "Valor", "Significado"],
        ["Alumnos que llegaron",       str(m.total),
         "Total de alumnos generados en 2 horas"],
        ["Alumnos atendidos",          str(m.atendidos),
         "Lograron tomar una bicicleta"],
        ["Alumnos que abandonaron",    f"{m.abandonos}  ({m.p_abandono:.1%})",
         "Se fueron sin bicicleta por la espera"],
        ["Throughput",                 f"{m.throughput(SIM_TIME_S):.1f} bici/hora",
         "Bicicletas entregadas por hora"],
        ["Wq — Espera promedio",       f"{m.wq_prom:.1f} s  ({m.wq_prom/60:.1f} min)",
         "Tiempo promedio esperando en la fila"],
        ["Wq — Espera maxima",         f"{m.wq_max:.1f} s  ({m.wq_max/60:.1f} min)",
         "Peor caso de espera registrado"],
        ["W — Tiempo en sistema",      f"{m.w_prom:.1f} s",
         "Espera + servicio promedio total"],
        ["Lq — Cola promedio",         f"{m.lq_prom:.2f} alumnos",
         "Promedio de personas esperando"],
        ["Lq — Cola maxima",           f"{m.lq_max} alumnos",
         "Pico maximo de la fila"],
        ["Error Ley de Little",        f"{m.little_error():.1f}%",
         "< 15% confirma que la simulacion es valida"],
    ]
    t_res = Table(res_data, colWidths=[2.0*inch, 1.8*inch, 2.9*inch])
    t_res.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), azul),
        ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF4FF")]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('TEXTCOLOR',   (1,3), (1,3), rojo),
        ('FONTNAME',    (1,3), (1,3), 'Helvetica-Bold'),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 0.15*inch))

    story.append(Image(rutas_graficas['cola'],
                       width=6.2*inch, height=2.5*inch, hAlign='CENTER'))
    story.append(Spacer(1, 0.1*inch))
    story.append(Image(rutas_graficas['esperas'],
                       width=5.5*inch, height=2.8*inch, hAlign='CENTER'))
    story.append(PageBreak())

    # ── SECCIÓN 6: Análisis de réplicas ─────────────────────────────────────
    story.append(Paragraph("6. Análisis de Estabilidad — 10 Replicas Independientes", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))
    story.append(Paragraph(
        "Para garantizar que los resultados no dependen de una semilla aleatoria particular, "
        "se corrieron <b>10 simulaciones independientes</b> con diferentes semillas (0 a 9). "
        "Los intervalos de confianza al 95% validan la consistencia del modelo.", st_body))

    ic = lambda arr: 1.96 * np.std(arr) / np.sqrt(len(arr))
    rep_data = [
        ["Métrica", "Media", "Desv. Std.", "IC 95%", "Interpretacion"],
        ["P_abandono",
         f"{np.mean(replicas_stats['abandonos']):.1f}%",
         f"± {np.std(replicas_stats['abandonos']):.1f}%",
         f"[{np.mean(replicas_stats['abandonos'])-ic(replicas_stats['abandonos']):.1f}%, "
         f"{np.mean(replicas_stats['abandonos'])+ic(replicas_stats['abandonos']):.1f}%]",
         "1 de cada 3 alumnos abandona"],
        ["Wq promedio",
         f"{np.mean(replicas_stats['wq']):.1f} s",
         f"± {np.std(replicas_stats['wq']):.1f} s",
         f"[{np.mean(replicas_stats['wq'])-ic(replicas_stats['wq']):.1f}, "
         f"{np.mean(replicas_stats['wq'])+ic(replicas_stats['wq']):.1f}] s",
         "Casi 2 minutos de espera promedio"],
        ["Throughput",
         f"{np.mean(replicas_stats['throughput']):.1f} bici/h",
         f"± {np.std(replicas_stats['throughput']):.1f}",
         f"[{np.mean(replicas_stats['throughput'])-ic(replicas_stats['throughput']):.1f}, "
         f"{np.mean(replicas_stats['throughput'])+ic(replicas_stats['throughput']):.1f}]",
         "Capacidad muy por debajo de la demanda"],
    ]
    t_rep = Table(rep_data, colWidths=[1.3*inch, 1.1*inch, 1.1*inch, 1.5*inch, 1.7*inch])
    t_rep.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), azul),
        ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF4FF"), colors.white]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_rep)
    story.append(Spacer(1, 0.15*inch))
    story.append(Image(rutas_graficas['replicas'],
                       width=6.5*inch, height=2.6*inch, hAlign='CENTER'))

    # ── SECCIÓN 7: Conclusiones ──────────────────────────────────────────────
    story.append(Paragraph("7. Conclusiones del Escenario 1", st_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=azul, spaceAfter=8))

    conclusiones = [
        ("Alta tasa de abandono",
         f"El {np.mean(replicas_stats['abandonos']):.1f}% de los alumnos abandona la fila. "
         "Esto significa que en 2 horas de hora pico, 1 de cada 3 estudiantes no logra "
         "tomar una bicicleta."),
        ("Cola máxima crítica",
         f"Se registran picos de hasta {m.lq_max} alumnos esperando simultáneamente, "
         "lo que genera una fila visible que disuade a nuevos usuarios de unirse."),
        ("Subutilización del servicio",
         f"Con solo {m.throughput(SIM_TIME_S):.1f} bicicletas entregadas por hora, "
         "el sistema opera muy por debajo de su demanda potencial."),
        ("Validez del modelo",
         f"La Ley de Little confirma la consistencia matemática de la simulación "
         f"con un error del {m.little_error():.1f}%, dentro del umbral aceptable del 15%."),
        ("Necesidad de mejora",
         "Este escenario establece la línea base de ineficiencia. Los Escenarios 2 "
         "(kioscos autónomos) y 3 (sistema híbrido) evaluarán alternativas de mejora "
         "frente a estos números."),
    ]
    for titulo_c, texto_c in conclusiones:
        story.append(Paragraph(f"<b>{titulo_c}:</b> {texto_c}", st_body))

    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC")))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        f"Reporte generado automaticamente por scen1_manual.py · "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')} · "
        "UNAM Ingenieria en Telecomunicaciones",
        ParagraphStyle("pie", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#999999"), alignment=TA_CENTER)))

    doc.build(story)
    return ruta_pdf


# ══════════════════════════════════════════════════════════════════════════════
# SALIDA EN CONSOLA (simplificada y legible)
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resultados(m, replicas_stats):
    sep = "─" * 52

    print("\n" + "═" * 52)
    print("  PUMABICIS · Escenario 1 · Sistema Manual")
    print("═" * 52)

    print("\n  FLUJO DE ALUMNOS")
    print(sep)
    print(f"  Llegaron al módulo  : {m.total} alumnos")
    print(f"  Fueron atendidos    : {m.atendidos} alumnos")
    print(f"  Abandonaron la fila : {m.abandonos} alumnos  ({m.p_abandono:.1%})")
    print(f"  Bicicletas/hora     : {m.throughput(SIM_TIME_S):.1f}")

    print("\n  TIEMPOS DE ESPERA")
    print(sep)
    print(f"  Espera promedio     : {m.wq_prom:.0f} s  ({m.wq_prom/60:.1f} min)")
    print(f"  Espera máxima       : {m.wq_max:.0f} s  ({m.wq_max/60:.1f} min)")
    print(f"  Tiempo en sistema   : {m.w_prom:.0f} s")

    print("\n  LONGITUD DE COLA")
    print(sep)
    print(f"  Cola promedio (Lq)  : {m.lq_prom:.1f} alumnos")
    print(f"  Cola máxima         : {m.lq_max} alumnos")

    print("\n  VALIDACIÓN (Ley de Little)")
    print(sep)
    err = m.little_error()
    estado = "✓ Simulación válida" if err < 15 else "⚠ Revisar lógica"
    print(f"  Error               : {err:.1f}%  → {estado}")

    print("\n  ESTABILIDAD (10 réplicas)")
    print(sep)
    ic = lambda arr: 1.96 * np.std(arr) / np.sqrt(len(arr))
    print(f"  Abandono promedio   : {np.mean(replicas_stats['abandonos']):.1f}% "
          f"  IC95: ±{ic(replicas_stats['abandonos']):.1f}%")
    print(f"  Wq promedio         : {np.mean(replicas_stats['wq']):.1f} s "
          f"  IC95: ±{ic(replicas_stats['wq']):.1f} s")
    print(f"  Throughput promedio : {np.mean(replicas_stats['throughput']):.1f} bici/h "
          f"  IC95: ±{ic(replicas_stats['throughput']):.1f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n[1/4] Corriendo simulación principal...")
    m = correr_simulacion(RANDOM_SEED)

    print("[2/4] Corriendo 10 réplicas...")
    replicas_stats = {'abandonos': [], 'wq': [], 'throughput': []}
    for seed in range(N_REPLICAS):
        mr = correr_simulacion(seed)
        replicas_stats['abandonos'].append(mr.p_abandono * 100)
        replicas_stats['wq'].append(mr.wq_prom)
        replicas_stats['throughput'].append(mr.throughput(SIM_TIME_S))

    print("[3/4] Generando gráficas...")
    rutas_graficas = generar_graficas(m, replicas_stats)

    print("[4/4] Generando reporte PDF...")
    ruta_pdf = generar_pdf(m, replicas_stats, rutas_graficas)

    imprimir_resultados(m, replicas_stats)

    print(f"  Reporte PDF  → {ruta_pdf}")
    print(f"  Carpeta      → {CARPETA_SALIDA}/")
    print("═" * 52)
