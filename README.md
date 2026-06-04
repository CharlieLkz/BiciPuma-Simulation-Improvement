# BiciPuma-Simulation-Improvement
Simulación usando Teoría de Colas y optimización del servicio usando SimPy para el sistema BiciPuma en la UNAM

# 🚲 Simulación usando Teoría de Colas: BiciPuma

Este proyecto implementa una simulación basada en eventos discretos utilizando **SimPy** en Python, con el objetivo de modelar, analizar y proponer mejoras eficientes para el sistema de préstamo de bicicletas en Ciudad Universitaria.

## 📌 Contexto del Problema
Durante las horas pico (05:45 - 07:45 hrs), las cicloestaciones experimentan **llegadas en ráfaga** provocadas por la descarga de pasajeros de los trenes (grupos de 1 a 8 personas). Actualmente, la validación manual (revisión de NIP, resello, casco y código de barras) genera un cuello de botella con tiempos de servicio elevados, resultando en largas colas y altas tasas de abandono.

## 🎯 Objetivos de la Simulación
1. **Modelar el Sistema Actual (M/M/1/K modificado):** Mapear la eficiencia del servidor manual.
2. **Proponer un Sistema Autónomo:** Simular la implementación de Kioscos RFID/QR.
3. **Validar un Sistema Híbrido:** Demostrar matemáticamente la viabilidad de un ecosistema que combina el autoservicio masivo con atención humana exclusiva para excepciones.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.x
* **Librería de Simulación:** SimPy
* **Control de Versiones:** Git / GitHub

## 🔀 Estructura de Ramas (Branches)
El desarrollo está dividido en escenarios comparativos:
* `main`: Código final y documentación.
* `scen-1-manual`: Modelado del sistema actual.
* `scen-2-autonomo`: Modelado de kioscos de autoservicio al 100%.
* `scen-3-hibrido`: Modelo mixto (Autoservicio + Ventanilla de excepciones).

---
*Proyecto desarrollado por Carlos Alberto Lazcano Vásquez - Facultad de Ingeniería, UNAM.*
