import numpy as np

# Definimos la función a integrar
def funcion(x):
    return (np.e)**x**2  # Puedes cambiarla por otra función

# Límites de integración
a = 0  # Límite inferior
b = 2  # Límite superior

# Número de subintervalos
n = 10 # Ajusta según la precisión deseada

# Paso
h = (b - a) / n

# Puntos medios
x_medio = np.linspace(a + h / 2, b - h / 2, n)

# Aplicamos la regla del rectángulo medio compuesta
integral = h * np.sum(funcion(x_medio))

print(f"Integral aproximada con la regla del rectángulo medio compuesta: {integral:.4f}")