import numpy as np

# Definir la función
def f(x):
    return np.exp(x) * np.sin(x)

# Punto y paso
x = 1
h = 0.01

# Diferencias finitas centradas para la primera derivada
def primera_derivada(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Diferencias finitas centradas para la segunda derivada
def segunda_derivada(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

# Calcular las derivadas
primera = primera_derivada(f, x, h)
segunda = segunda_derivada(f, x, h)
error_absoluto_primera = (abs(f(x) - primera))
error_absoluto_segunda = (abs(f(x) - segunda))
error_relativo_primera = (error_absoluto_primera / abs(f(x))) * 100
error_relativo_segunda = (error_absoluto_segunda / abs(f(x))) * 100 

# Imprimir los resultados
print(f"Primera derivada en x = {x}: {primera}")
print(f"Segunda derivada en x = {x}: {segunda}")
print(f"Error absoluto primera derivada: {error_absoluto_primera}")
print(f"Error absoluto segunda derivada: {error_absoluto_segunda}")
print(f"Error relativo primera derivada: {error_relativo_primera:.2f}%")
print(f"Error relativo segunda derivada: {error_relativo_segunda:.2f}%")