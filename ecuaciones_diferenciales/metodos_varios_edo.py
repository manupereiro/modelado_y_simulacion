import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
def euler(f, y0, t0, tf, h):
    """ Método de Euler estándar. """
    t_vals = np.arange(t0, tf + h, h)
    y_vals = np.zeros_like(t_vals)
    y_vals[0] = y0
 
    for i in range(len(t_vals) - 1):
        y_vals[i + 1] = y_vals[i] + h * f(t_vals[i], y_vals[i])
 
    return t_vals, y_vals
 
def euler_mejorado(f, y0, t0, tf, h):
    """ Método de Euler mejorado (Heun). """
    t_vals = np.arange(t0, tf + h, h)
    y_vals = np.zeros_like(t_vals)
    y_vals[0] = y0
 
    for i in range(len(t_vals) - 1):
        t, y = t_vals[i], y_vals[i]
        y_pred = y + h * f(t, y)  # Predicción con Euler
        y_vals[i + 1] = y + h * (f(t, y) + f(t + h, y_pred)) / 2  # Corrección
 
    return t_vals, y_vals
 
def runge_kutta_4(f, y0, t0, tf, h):
    """ Método de Runge-Kutta de cuarto orden (RK4). """
    t_vals = np.arange(t0, tf + h, h)
    y_vals = np.zeros_like(t_vals)
    y_vals[0] = y0
 
    for i in range(len(t_vals) - 1):
        t, y = t_vals[i], y_vals[i]
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)
        y_vals[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6  # Fórmula RK4
 
    return t_vals, y_vals
 
# Nueva ecuación diferencial: dy/dt = t + y
def f(t, y):
    return y * np.sin(t)
 
# Solución exacta usando método analítico: y(t) = e^t - t - 1
def exact_solution(y0, t_vals):
    return np.exp(-np.cos(t_vals) + 1)
 
# Parámetros
y0 = 1      # Condición inicial
t0 = 0      # Tiempo inicial
tf = np.pi      # Tiempo final
h = np.pi/10     # Paso de integración
 
# Cálculo de soluciones
t_vals, y_euler = euler(f, y0, t0, tf, h)
_, y_euler_mejorado = euler_mejorado(f, y0, t0, tf, h)
_, y_rk = runge_kutta_4(f, y0, t0, tf, h)
y_exact = exact_solution(y0, t_vals)
 
# Crear tabla resumen
tabla_resumen = pd.DataFrame({
    'Tiempo (t)': t_vals, 
    'Exacta': y_exact, 
    'Euler': y_euler, 
    'Euler Mejorado': y_euler_mejorado, 
    'Runge-Kutta 4': y_rk
})
 
print(tabla_resumen)
 
# Graficamos los resultados con líneas diferenciadas
plt.figure(figsize=(8, 5))
plt.plot(t_vals, y_exact, label="Solución Exacta", color="blue", linestyle="-", linewidth=2)
plt.plot(t_vals, y_euler, label="Euler", color="orange", linestyle="dotted", linewidth=2, marker="s")
plt.plot(t_vals, y_euler_mejorado, label="Euler Mejorado", color="green", linestyle="dashdot", linewidth=2, marker="d")
plt.plot(t_vals, y_rk, label="Runge-Kutta 4", color="red", linestyle="--", linewidth=2, marker="o")
 
plt.xlabel("Tiempo (t)")
plt.ylabel("Solución (y)")
plt.title("Comparación de métodos numéricos para EDO: dy/dt = t + y")
plt.legend()
plt.grid()
plt.show()