import numpy as np
from scipy.integrate import quad # Para obtener un valor de referencia "exacto"

# --- Parámetros Configurables ---
precision_digitos = 6 # Número de dígitos decimales para mostrar en los resultados

# --- Funciones ---
def funcion(x):
    """Define la función a integrar."""
    return np.sqrt(2)*np.exp(x**2)  # Puedes cambiarla por cualquier otra función

def regla_simpson_13_compuesta(f, a, b, n):
    """Calcula la integral usando la regla de Simpson 1/3 compuesta."""
    if n <= 0:
        raise ValueError("El número de subintervalos 'n' debe ser positivo.")
    if n % 2 != 0:
        raise ValueError("El número de subintervalos 'n' debe ser par para Simpson 1/3.")
        
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # Aplicamos la regla de Simpson 1/3 compuesta
    S = y[0] + y[-1] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2])
    integral = (h / 3) * S
    return integral

# --- Ejecución Principal ---

# Límites de integración
a = 0          # Límite inferior
b = 1  # Límite superior

# Número de subintervalos (debe ser par)
n = 4         # Ajusta según la precisión deseada (asegúrate de que sea par)
if n % 2 != 0:
    print(f"Advertencia: n={n} no es par. Incrementando a n={n+1} para Simpson 1/3.")
    n += 1

# 1. Calcular la integral aproximada
integral_aproximada = regla_simpson_13_compuesta(funcion, a, b, n)

# 2. Obtener un valor de referencia "exacto" usando SciPy
valor_exacto, error_estimado_quad = quad(funcion, a, b)

# 3. Calcular errores
error_absoluto = abs(valor_exacto - integral_aproximada)

# Evitar división por cero si el valor exacto es 0
if abs(valor_exacto) < 1e-15: # Usar una tolerancia pequeña
    error_relativo_porcentaje = float('inf') if error_absoluto > 1e-15 else 0.0
else:
    error_relativo_porcentaje = (error_absoluto / abs(valor_exacto)) * 100

# 4. Imprimir resultados con la precisión deseada
print(f"--- Resultados (Precisión: {precision_digitos} dígitos) ---")
print(f"Límites de integración: [{a}, {b}]")
print(f"Número de subintervalos (n): {n}")
print(f"Integral aproximada (Simpson 1/3 Compuesta): {integral_aproximada:.{precision_digitos}f}")
print(f"Valor de referencia (SciPy quad):            {valor_exacto:.{precision_digitos}f}")
print(f"Error Absoluto:                              {error_absoluto:.{precision_digitos}f}")
print(f"Error Relativo Porcentual:                   {error_relativo_porcentaje:.{precision_digitos}f}%")