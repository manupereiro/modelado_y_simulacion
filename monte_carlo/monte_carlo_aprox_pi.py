import numpy as np
 
# Fijar la semilla para reproducibilidad
np.random.seed(42)
 
# Parámetros del círculo
radio = 1
centro_x, centro_y = 0, 0  # El círculo está centrado en el origen
 
# Número de puntos aleatorios
n = 10000
 
# Generar coordenadas aleatorias en el intervalo [-1,1] para x y y
x_random = np.random.uniform(-radio, radio, n)
y_random = np.random.uniform(-radio, radio, n)
 
# Verificar si los puntos están dentro del círculo (x^2 + y^2 ≤ radio^2)
dentro_circulo = x_random**2 + y_random**2 <= radio**2
 
# Contar cuántos puntos están dentro del círculo
num_exitos = np.sum(dentro_circulo)
 
print(f"Número de puntos dentro del círculo: {num_exitos} de {n}")