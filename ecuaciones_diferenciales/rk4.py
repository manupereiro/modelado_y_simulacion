import numpy as np
import pandas as pd
 
def f(t, y):
    """Función de la ecuación diferencial: dy/dt = t - y**2."""
    return y * np.sin(t) 
 
def runge_kutta_4(f, y0, t0, tf, h):
    """Método de Runge-Kutta de cuarto orden (RK4) con pendientes correctamente en la tabla."""
    t_values = np.arange(t0, tf + h, h)
    y_values = [y0]
    tabla_resumen = []
 
    for i in range(len(t_values) - 1):
        t_n = t_values[i]
        y_n = y_values[-1]
 
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n +h* k1/2)
        k3 = f(t_n + h/2, y_n +h* k2/2)
        k4 = f(t_n + h, y_n +h*k3)
 
        y_next = y_n + (k1 + 2*k2 + 2*k3 + k4)*h / 6
        y_values.append(y_next)
 
        tabla_resumen.append({
            "Iteración": i+1,
            "Tiempo": round(t_n, 4),
            "y_actual": round(y_n, 6),
            "k1": round(k1, 6),
            "k2": round(k2, 6),
            "k3": round(k3, 6),
            "k4": round(k4, 6),
            "y_siguiente": round(y_next, 6)
        })
 
    return t_values, y_values, tabla_resumen
 
# Parámetros
y0 = 1   # Condición inicial
t0 = 0   # Tiempo inicial
tf = np.pi  # Tiempo final
h = np.pi/10  # Paso de integración
 
# Cálculo de la solución con RK4
t_vals, y_vals, resumen = runge_kutta_4(f, y0, t0, tf, h)
 
# ** Ajustar el ancho de la tabla antes de mostrarla **
pd.options.display.width = 150  # Aumenta el ancho de visualización
pd.options.display.max_columns = 10  # Asegura que todas las columnas sean visibles
 
# Mostrar tabla resumen con las pendientes correctamente formateadas
tabla_df = pd.DataFrame(resumen)
print(tabla_df)