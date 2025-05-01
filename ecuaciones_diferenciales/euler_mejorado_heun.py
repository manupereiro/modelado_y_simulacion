import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def euler_mejorado(f, y0, t0, tf, h):
    """ Método de Euler mejorado (Heun). """
    t = np.arange(t0, tf + h, h)
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(len(t) - 1):
        y_pred = y[i] + h * f(t[i], y[i])  # Predicción con Euler
        # Corrección
        y[i + 1] = y[i] + h * (f(t[i], y[i]) + f(t[i] + h, y_pred)) / 2
    return t, y

# Ecuación diferencial: dy/dt = 0.4 * t * y
def f(t, y):
    return y * np.sin(t) 

# Solución exacta
def sol_exacta(y0, t):
    # Para y0=1, t0=1, la constante es C = 1/exp(0.2)
    return np.exp(-np.cos(t) + 1)

# Parámetros
y0, t0, tf, h = 1, 0, np.pi, np.pi/10 # Condiciones iniciales

# Cálculo de soluciones
t, y_eul_mej = euler_mejorado(f, y0, t0, tf, h)
y_real = sol_exacta(y0, t) # y0 no se usa realmente en esta versión específica de sol_exacta

# Crear tabla resumen
tabla = pd.DataFrame({
    'Tiempo': t,
    'Exacta': y_real,
    'Euler Mejorado': y_eul_mej
})
print(tabla)

# Código para Heun (El método implementado es Heun)

# Graficar resultados
plt.figure(figsize=(10, 6)) # Ajustar tamaño para mejor visualización
plt.plot(t, y_real, label="Solución Exacta", color="blue", linestyle="-", linewidth=2)
plt.plot(t, y_eul_mej, label="Euler Mejorado (Heun)", color="green", linestyle="--",
         linewidth=1.5, marker="o", markersize=4) # Ajustar estilo para diferenciar
plt.xlabel("Tiempo (t)")
plt.ylabel("Solución y(t)")
plt.title("Comparación: Euler Mejorado (Heun) vs Solución Exacta")
plt.legend()
plt.grid(True) # Mantener el grid
plt.show()