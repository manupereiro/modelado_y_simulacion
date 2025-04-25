import numpy as np

# Definimos la función a integrar
def funcion(x):
    return np.sin(x)  # Puedes cambiarla por otra función

# Límites de integración
a = 0  # Límite inferior
b = np.pi  # Límite superior

# Paso
h = (b - a) / 3

# Puntos de evaluación
x1 = a + h
x2 = a + 2 * h

# Evaluamos la función en los puntos
fa = funcion(a)
fx1 = funcion(x1)
fx2 = funcion(x2)
fb = funcion(b)

# Aplicamos la regla de Simpson 3/8 simple
integral = (3 * h / 8) * (fa + 3 * fx1 + 3 * fx2 + fb)

print(f"Integral aproximada con Simpson 3/8 simple: {integral:.4f}")