import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib.patches import Circle

# Parámetros
n_points_simulation = 10000 # Número de puntos para la simulación/visualización
n_points_estimation = 10000 # Número de puntos para la estimación numérica
n_experiments = 10
seed_value = 0 # Semilla para reproducibilidad

# Establecer la semilla para numpy
np.random.seed(seed_value)

def estimate_pi_and_get_points(n):
    """Estima Pi y devuelve los puntos generados."""
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside) / n
    return pi_estimate, x, y, inside

# --- Estimación Numérica (como antes) ---
print(f"--- Estimación de Pi ({n_experiments} experimentos de {n_points_estimation} puntos c/u) ---")
print(f"Usando semilla aleatoria: {seed_value}")
results = [estimate_pi_and_get_points(n_points_estimation)[0] for _ in range(n_experiments)]
prom = np.mean(results)
table = [[i+1, r] for i,r in enumerate(results)]
table.append(["Promedio", prom])
print(tabulate(table, headers=["Experimento", "Estimación π"], tablefmt="grid", floatfmt=".6f"))
print("-" * 50)

# --- Visualización del Método Monte Carlo ---
print(f"\n--- Generando gráfico con {n_points_simulation} puntos ---")
# Ejecutar una vez para obtener puntos para graficar
# Reiniciar la semilla si quieres que el gráfico use la misma secuencia inicial que el primer experimento
# np.random.seed(seed_value)
pi_est_graph, x_pts, y_pts, inside_pts = estimate_pi_and_get_points(n_points_simulation)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(8, 8))

# Dibujar el círculo unitario
circle = Circle((0, 0), 1, edgecolor='black', facecolor='none', linewidth=2, label='Círculo (x²+y²=1)')
ax.add_patch(circle)

# Dibujar el cuadrado
square = plt.Rectangle((-1, -1), 2, 2, edgecolor='black', facecolor='none', linestyle='--', label='Cuadrado [-1,1]x[-1,1]')
ax.add_patch(square)

# Separar puntos dentro y fuera
x_inside = x_pts[inside_pts]
y_inside = y_pts[inside_pts]
x_outside = x_pts[~inside_pts]
y_outside = y_pts[~inside_pts]

# Graficar los puntos
ax.scatter(x_inside, y_inside, color='blue', s=5, label=f'Dentro ({len(x_inside)} puntos)')
ax.scatter(x_outside, y_outside, color='red', s=5, label=f'Fuera ({len(x_outside)} puntos)')

# Configuración del gráfico
ax.set_aspect('equal', adjustable='box') # Asegura que el círculo se vea como un círculo
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Método Monte Carlo para estimar π ({n_points_simulation} puntos)\nEstimación para este gráfico: π ≈ {pi_est_graph:.4f}')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.6)

# Mostrar el gráfico
plt.show()