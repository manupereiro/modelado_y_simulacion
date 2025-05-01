import numpy as np
from scipy.integrate import quad # Para obtener un valor de referencia "exacto"

# --- Parámetros Configurables ---
precision_digitos = 6 # Número de dígitos decimales para mostrar en los resultados

# --- Funciones ---
def funcion(x):
    """Define la función a integrar."""
    return np.exp(x**2) # Función original (integral no elemental)

def regla_rectangulo_compuesta(f, a, b, n):
    """Calcula la integral usando la regla del rectángulo medio compuesta."""
    if n <= 0:
        raise ValueError("El número de subintervalos 'n' debe ser positivo.")
    h = (b - a) / n
    x_medio = np.linspace(a + h / 2, b - h / 2, n)
    integral = h * np.sum(f(x_medio))
    return integral

# --- Ejecución Principal ---

# Límites de integración
a = 0  # Límite inferior
b = 2  # Límite superior (Cambiado a 1 para que quad converja más fácilmente)

# Número de subintervalos
n = 20 # Ajusta según la precisión deseada

# 1. Calcular la integral aproximada
integral_aproximada = regla_rectangulo_compuesta(funcion, a, b, n)

# 2. Obtener un valor de referencia "exacto" usando SciPy
#    Nota: La integral de exp(x^2) no tiene solución elemental. quad da una aproximación muy precisa.
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
print(f"Integral aproximada (Rectángulo Compuesto): {integral_aproximada:.{precision_digitos}f}")
print(f"Valor de referencia (SciPy quad):           {valor_exacto:.{precision_digitos}f}")
print(f"Error Absoluto:                             {error_absoluto:.{precision_digitos}f}")
print(f"Error Relativo Porcentual:                  {error_relativo_porcentaje:.{precision_digitos}f}%")