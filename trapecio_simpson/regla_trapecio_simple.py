import numpy as np

# Definimos la función a integrar
def funcion(x):
    return np.sin(x)  # Puedes cambiarla por cualquier otra función

# Límites de integración
a = 0  # Límite inferior
b = np.pi / 2  # Límite superior

# Evaluamos la función en los extremos
fa = funcion(a)
fb = funcion(b)

# Aplicamos la regla del trapecio simple
integral = (b - a) / 2 * (fa + fb)

print(f"Integral aproximada con la regla del trapecio simple: {integral:.4f}")