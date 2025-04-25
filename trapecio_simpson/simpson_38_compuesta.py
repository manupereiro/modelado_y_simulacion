import numpy as np

# Definimos la función a integrar
def funcion(x):
    return 6 + 3 * np.cos(x)  # Puedes cambiarla por otra función

# Límites de integración
a = 0  # Límite inferior
b = np.pi/2  # Límite superior

# Número de subintervalos (debe ser múltiplo de 3)
n = 6  # Ajusta según la precisión deseada

# Paso
h = (b - a) / n

# Puntos de evaluación
x = np.linspace(a, b, n + 1)
y = funcion(x)

# Aplicamos la regla de Simpson 3/8 compuesta
S = y[0] + y[-1] + 3 * np.sum(y[1:n:3]) + 3 * np.sum(y[2:n:3]) + 2 * np.sum(y[3:n-1:3])
integral = (3 * h / 8) * S

print(f"Integral aproximada con la regla de Simpson 3/8: {integral:.4f}")