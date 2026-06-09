"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PUMABICIS — Escenario 2: Sistema 100% Autónomo (rama: scen-2-autonomo)   ║
║  3 Kioscos RFID/QR · Falla de bloqueo con P=5%                            ║
║  UNAM · Ingeniería en Telecomunicaciones · Teoría de Colas                ║
╚══════════════════════════════════════════════════════════════════════════════╝

MODELO: M[X] / U(10,20) / 3 / ∞ + BLOQUEO(P=0.05, t=3–5 min) + RENEGING

DIFERENCIA CLAVE vs Escenario 1:
  • 3 kioscos en paralelo → capacidad 3x mayor
  • Servicio rápido: Uniforme(10, 20) s por alumno
  • PERO: el 5% de alumnos provoca un bloqueo del kiosco completo
    → el kiosco queda FUERA DE SERVICIO 3–5 min (Uniforme(180, 300) s)
    → los demás alumnos en cola deben esperar con menos recursos
  • Sin soporte humano → los alumnos bloqueados SIEMPRE abandonan
"""

import simpy
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os

# ══════════════════════════════════════════════════════════════════════════════
# 1. PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

# Llegadas (iguales al Escenario 1 — misma realidad de campo)
INTER_LLEGADAS_MIN = [5, 3, 7, 5, 10, 3, 7, 5, 3, 7, 10]
PERSONAS_POR_TREN  = [4, 7, 1, 3, 8, 4, 2, 5, 4, 3, 1, 8]

LAMBDA_TRENES_S = 1 / (np.mean(INTER_LLEGADAS_MIN) * 60)  # ~0.00282 /s
MEDIA_LOTE      = np.mean(PERSONAS_POR_TREN)               # ~4.17
LAMBDA_EFF_S    = LAMBDA_TRENES_S * MEDIA_LOTE             # ~0.01176 /s

# Kioscos
N_KIOSCOS        = 3
SRV_MIN_S        = 10    # mínimo de servicio normal (s)
SRV_MAX_S        = 20    # máximo de servicio normal (s)

# Falla / bloqueo
P_EXCEPCION      = 0.05  # probabilidad de que un alumno provoque bloqueo
BLOQUEO_MIN_S    = 180   # 3 minutos de bloqueo mínimo
BLOQUEO_MAX_S    = 300   # 5 minutos de bloqueo máximo

# Paciencia del alumno
PATIENCE_MEAN_S  = 300   # media exponencial de paciencia (5 min)

SIM_TIME_S       = 7200  # 2 horas de hora pico
RANDOM_SEED      = 42


# ══════════════════════════════════════════════════════════════════════════════
# 2. ENTIDAD Y MÉTRICAS
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
        self.fue_excepcion = False  # ¿causó un bloqueo?

    @property
    def t_espera(self):
        return (self.t_inicio_srv - self.t_llegada) if self.t_inicio_srv else 0.0

    @property
    def t_sistema(self):
        return (self.t_salida - self.t_llegada) if (self.t_salida and self.atendido) else None


class Metricas:
    def __init__(self):
        self.alumnos:      list[Alumno]         = []
        self.log_cola:     list[tuple[float,int]] = []
        self.log_kioscos:  list[tuple[float,int]] = []  # kioscos activos en el tiempo
        self.bloqueos:     list[dict]             = []  # registro de cada bloqueo

    def registrar(self, a): self.alumnos.append(a)
    def log_lq(self, t, lq): self.log_cola.append((t, lq))
    def log_activos(self, t, n): self.log_kioscos.append((t, n))
    def registrar_bloqueo(self, kiosco_id, t_inicio, t_fin):
        self.bloqueos.append({"kiosco": kiosco_id, "inicio": t_inicio,
                              "fin": t_fin, "duracion": t_fin - t_inicio})

    @property
    def total(self):       return len(self.alumnos)
    @property
    def atendidos(self):   return sum(1 for a in self.alumnos if a.atendido)
    @property
    def abandonos(self):   return sum(1 for a in self.alumnos if a.abandono)
    @property
    def excepciones(self): return sum(1 for a in self.alumnos if a.fue_excepcion)

    @property
    def p_abandono(self):
        return self.abandonos / self.total if self.total else 0.0

    @property
    def wq_promedio(self):
        t = [a.t_espera for a in self.alumnos if a.atendido]
        return np.mean(t) if t else 0.0

    @property
    def wq_max(self):
        t = [a.t_espera for a in self.alumnos if a.atendido]
        return max(t) if t else 0.0

    @property
    def w_promedio(self):
        t = [a.t_sistema for a in self.alumnos if a.t_sistema]
        return np.mean(t) if t else 0.0

    @property
    def lq_promedio(self):
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

    def verificar_little(self, lambda_eff):
        lq_l   = lambda_eff * self.wq_promedio
        lq_emp = self.lq_promedio
        err    = abs(lq_l - lq_emp) / (lq_emp + 1e-9) * 100
        return {"Lq_little": lq_l, "Lq_emp": lq_emp,
                "error_%": err, "valido": err < 20.0}

    def resumen(self, sim_time, lambda_eff):
        lt = self.verificar_little(lambda_eff)
        s  = "✓" if lt["valido"] else "⚠"
        tiempo_bloqueado = sum(b["duracion"] for b in self.bloqueos)
        print("\n" + "=" * 65)
        print("  RESULTADOS — Escenario 2: Sistema 100% autónomo")
        print("=" * 65)
        print(f"  Alumnos llegados    : {self.total}")
        print(f"  Alumnos atendidos   : {self.atendidos}")
        print(f"  Abandonos           : {self.abandonos}  ({self.p_abandono:.1%})")
        print(f"  Excepciones (5%)    : {self.excepciones}  (generaron bloqueo)")
        print(f"  Bloqueos totales    : {len(self.bloqueos)}")
        print(f"  Tiempo bloqueado    : {tiempo_bloqueado:.0f} s  "
              f"({tiempo_bloqueado/60:.1f} min de {sim_time/60:.0f} min totales)")
        print(f"  Throughput          : {self.throughput(sim_time):.1f} bici/hora")
        print()
        print(f"  Wq promedio (espera): {self.wq_promedio:.1f} s  ({self.wq_promedio/60:.2f} min)")
        print(f"  Wq máximo           : {self.wq_max:.1f} s  ({self.wq_max/60:.2f} min)")
        print(f"  W promedio (sistema): {self.w_promedio:.1f} s")
        print()
        print(f"  Lq promedio (cola)  : {self.lq_promedio:.2f} alumnos")
        print(f"  Lq máximo           : {self.lq_max} alumnos")
        print()
        print(f"  Ley de Little {s}")
        print(f"    Lq (L=λWq)        : {lt['Lq_little']:.3f}")
        print(f"    Lq (empírico)     : {lt['Lq_emp']:.3f}")
        print(f"    Error             : {lt['error_%']:.1f}%")
        print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RECURSO KIOSCO CON LÓGICA DE BLOQUEO
# ══════════════════════════════════════════════════════════════════════════════

class KioscoPool:
    """
    Envuelve simpy.Resource para modelar los bloqueos individuales.
    Cada kiosco tiene su propio estado (libre / ocupado / bloqueado).
    Usamos un solo Resource de capacity=N_KIOSCOS para la cola,
    pero cuando un kiosco entra en bloqueo, reducimos la capacidad
    disponible ocupando un slot con un proceso de bloqueo.
    """
    def __init__(self, env: simpy.Environment, n: int, metricas: Metricas):
        self.env      = env
        self.resource = simpy.Resource(env, capacity=n)
        self.n        = n
        self.metricas = metricas
        self._bloqueados = 0   # kioscos actualmente bloqueados

    @property
    def kioscos_activos(self):
        return self.n - self._bloqueados

    def bloquear(self, kiosco_id: int, duracion: float):
        """Proceso que simula un kiosco fuera de servicio."""
        self._bloqueados += 1
        t_ini = self.env.now
        self.metricas.log_activos(self.env.now, self.kioscos_activos)
        yield self.env.timeout(duracion)
        self._bloqueados -= 1
        self.metricas.registrar_bloqueo(kiosco_id, t_ini, self.env.now)
        self.metricas.log_activos(self.env.now, self.kioscos_activos)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PROCESOS SIMPY
# ══════════════════════════════════════════════════════════════════════════════

def proceso_alumno_scen2(env: simpy.Environment,
                          alumno: Alumno,
                          pool: KioscoPool,
                          metricas: Metricas,
                          rng: random.Random):
    """
    Proceso de un alumno en el sistema autónomo:
    1. Llega y espera un kiosco libre (o abandona si se agota paciencia)
    2. Si hay excepción (5%): el kiosco se bloquea Y el alumno abandona
       (sin soporte humano no hay forma de resolver la excepción)
    3. Si no hay excepción: servicio rápido Uniforme(10, 20) s
    """
    metricas.log_lq(env.now, len(pool.resource.queue))

    with pool.resource.request() as req:
        resultado = yield req | env.timeout(alumno.paciencia)

        if req not in resultado:
            # Abandono por impaciencia
            alumno.abandono = True
            alumno.t_salida = env.now
            metricas.registrar(alumno)
            metricas.log_lq(env.now, len(pool.resource.queue))
            return

        alumno.t_inicio_srv = env.now
        metricas.log_lq(env.now, len(pool.resource.queue))

        # ¿Ocurre una excepción?
        if rng.random() < P_EXCEPCION:
            # El kiosco se bloquea — lanzamos el proceso de bloqueo en paralelo
            # (el req sigue ocupado durante el bloqueo → el kiosco queda fuera)
            alumno.fue_excepcion = True
            duracion_bloqueo = rng.uniform(BLOQUEO_MIN_S, BLOQUEO_MAX_S)
            kiosco_id = rng.randint(1, N_KIOSCOS)  # ID simbólico para registro

            # El bloqueo ocupa el slot del kiosco durante la duración
            env.process(pool.bloquear(kiosco_id, duracion_bloqueo))
            yield env.timeout(duracion_bloqueo)  # el req se libera al fin del with

            # El alumno abandona: sin humano, no hay resolución
            alumno.abandono = True
            alumno.t_salida = env.now
            metricas.registrar(alumno)
            metricas.log_lq(env.now, len(pool.resource.queue))
        else:
            # Servicio normal rápido
            t_srv = rng.uniform(SRV_MIN_S, SRV_MAX_S)
            yield env.timeout(t_srv)
            alumno.atendido = True
            alumno.t_salida = env.now
            metricas.registrar(alumno)
            metricas.log_lq(env.now, len(pool.resource.queue))


def generador_trenes_scen2(env, pool, metricas, rng):
    """Mismo generador de llegadas en ráfaga que el Escenario 1."""
    while True:
        inter = rng.expovariate(LAMBDA_TRENES_S)
        yield env.timeout(inter)
        lote = rng.randint(1, 8)
        for _ in range(lote):
            paciencia = rng.expovariate(1 / PATIENCE_MEAN_S)
            alumno = Alumno(t_llegada=env.now, paciencia=paciencia)
            env.process(proceso_alumno_scen2(env, alumno, pool, metricas, rng))


# ══════════════════════════════════════════════════════════════════════════════
# 5. SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def correr_simulacion(semilla=RANDOM_SEED):
    Alumno._cnt = 0
    rng         = random.Random(semilla)
    env         = simpy.Environment()
    metricas    = Metricas()
    pool        = KioscoPool(env, N_KIOSCOS, metricas)

    env.process(generador_trenes_scen2(env, pool, metricas, rng))
    env.run(until=SIM_TIME_S)
    return metricas


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRAFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def graficar(m: Metricas, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pumabicis — Escenario 2: Sistema 100% Autónomo\n"
                 "3 Kioscos RFID/QR · P_excepción=5% · Bloqueo 3–5 min",
                 fontsize=13, fontweight="bold")

    # Cola en el tiempo
    ax = axes[0, 0]
    ts  = [x[0]/60 for x in m.log_cola]
    lqs = [x[1]    for x in m.log_cola]
    ax.step(ts, lqs, color="#E24B4A", linewidth=0.8, where='post')
    ax.axhline(m.lq_promedio, color="#185FA5", linestyle="--",
               label=f"Lq prom = {m.lq_promedio:.1f}")
    # Marcar bloqueoso
    for b in m.bloqueos:
        ax.axvspan(b["inicio"]/60, b["fin"]/60, color="#E06C00", alpha=0.2)
    ax.set_title("Cola en el tiempo (naranja = kiosco bloqueado)")
    ax.set_xlabel("Tiempo (min)"); ax.set_ylabel("Alumnos en cola")
    ax.legend(); ax.grid(alpha=0.3)

    # Histograma tiempos de espera
    ax = axes[0, 1]
    esperas = [a.t_espera for a in m.alumnos if a.atendido]
    ax.hist([e/60 for e in esperas], bins=20, color="#1D9E75",
            edgecolor="white")
    ax.axvline(m.wq_promedio/60, color="#E24B4A", linestyle="--",
               label=f"Wq prom = {m.wq_promedio/60:.2f} min")
    ax.set_title("Tiempos de espera (atendidos)")
    ax.set_xlabel("Espera (min)"); ax.set_ylabel("Frecuencia")
    ax.legend(); ax.grid(alpha=0.3)

    # Kioscos activos en el tiempo
    ax = axes[1, 0]
    if m.log_kioscos:
        tk = [x[0]/60 for x in m.log_kioscos]
        nk = [x[1]    for x in m.log_kioscos]
        ax.step(tk, nk, color="#185FA5", linewidth=1.5, where='post')
        ax.axhline(N_KIOSCOS, color="#1D9E75", linestyle="--",
                   label=f"Capacidad máx = {N_KIOSCOS}")
        ax.set_ylim(-0.2, N_KIOSCOS + 0.5)
    ax.set_title("Kioscos activos en el tiempo")
    ax.set_xlabel("Tiempo (min)"); ax.set_ylabel("Kioscos disponibles")
    ax.legend(); ax.grid(alpha=0.3)

    # Resumen
    ax = axes[1, 1]
    cats   = ["Llegados", "Atendidos", "Abandonos", "Bloqueados"]
    vals   = [m.total, m.atendidos, m.abandonos, len(m.bloqueos)]
    cols   = ["#185FA5", "#1D9E75", "#E24B4A", "#E06C00"]
    bars   = ax.bar(cats, vals, color=cols, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                str(v), ha="center", fontweight="bold")
    ax.set_title(f"Flujo · Abandono: {m.p_abandono:.1%}")
    ax.set_ylabel("Cantidad"); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    ruta = os.path.join(output_dir, "pumabicis_scen2_resultados.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] Gráfica guardada: {ruta}")


def exportar_csv(m: Metricas, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pumabicis_scen2_alumnos.csv"),
              "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","t_llegada_s","paciencia_s","t_espera_s",
                    "t_sistema_s","atendido","abandono","fue_excepcion"])
        for a in m.alumnos:
            w.writerow([a.id, round(a.t_llegada,2), round(a.paciencia,2),
                        round(a.t_espera,2) if a.atendido else "N/A",
                        round(a.t_sistema,2) if a.t_sistema else "N/A",
                        int(a.atendido), int(a.abandono), int(a.fue_excepcion)])
    print(f"[✓] CSV exportado: pumabicis_scen2_alumnos.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n╔" + "═"*63 + "╗")
    print("║  PUMABICIS · Escenario 2 · Sistema autónomo · SimPy        ║")
    print("╚" + "═"*63 + "╝")

    print("\n[→] Corriendo simulación (semilla=42, 7200 s)...")
    m = correr_simulacion(RANDOM_SEED)
    m.resumen(SIM_TIME_S, LAMBDA_EFF_S)

    output = "resultados_scen2"
    graficar(m, output)
    exportar_csv(m, output)

    print("\n[→] Réplicas (semillas 0–9)...")
    abandonos_r, wq_r, tp_r, bloqueos_r = [], [], [], []
    for seed in range(10):
        mr = correr_simulacion(seed)
        abandonos_r.append(mr.p_abandono * 100)
        wq_r.append(mr.wq_promedio)
        tp_r.append(mr.throughput(SIM_TIME_S))
        bloqueos_r.append(len(mr.bloqueos))

    print(f"\n  P_abandono  : {np.mean(abandonos_r):.1f}% ± {np.std(abandonos_r):.1f}%"
          f"  IC95≈[{np.mean(abandonos_r)-1.96*np.std(abandonos_r)/10**0.5:.1f}%, "
          f"{np.mean(abandonos_r)+1.96*np.std(abandonos_r)/10**0.5:.1f}%]")
    print(f"  Wq promedio : {np.mean(wq_r):.1f} s ± {np.std(wq_r):.1f} s")
    print(f"  Throughput  : {np.mean(tp_r):.1f} bici/h ± {np.std(tp_r):.1f}")
    print(f"  Bloqueos    : {np.mean(bloqueos_r):.1f} por turno ± {np.std(bloqueos_r):.1f}")
    print("\n[✓] Escenario 2 completo. Siguiente: scen-3-hibrido")
