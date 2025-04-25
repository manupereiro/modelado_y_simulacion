import numpy as np

# Definimos la función a integrar
def funcion(x):
    return 6 + 3 * np.cos(x)  # Puedes cambiarla por cualquier otra función

# Límites de integración
a = 0  # Límite inferior
b = (np.pi)/2  # Límite superior

# Número de subintervalos (debe ser par)
n = 4  # Ajusta según la precisión deseada

# Paso
h = (b - a) / n

# Puntos
x = np.linspace(a, b, n + 1)
y = funcion(x)

# Aplicamos la regla de Simpson compuesta
S = y[0] + y[-1] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n-1:2])
integral = (h / 3) * S

print(f"Integral aproximada con Simpson compuesta: {integral:.4f}")