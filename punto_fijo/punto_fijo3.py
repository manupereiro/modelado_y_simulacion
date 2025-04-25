import matplotlib.pyplot as plt
import numpy as np

def punto_fijo(g, x0, tol=1e-4, max_iter=100, mostrar_grafico=True):
    iteraciones = 0
    x_vals = [x0]
    errores = []

    print(f"{'Iter':<5} {'x_n':<15} {'g(x_n)':<15} {'Error':<15}")
    print("-" * 50)

    while iteraciones < max_iter:
        x_prev = x_vals[-1]
        x_next = g(x_prev)
        error = abs(x_next - x_prev)

        print(f"{iteraciones:<5} {x_prev:<15.10f} {x_next:<15.10f} {error:<15.10f}")
        errores.append(error)
        x_vals.append(x_next)

        if error < tol:
            break
        iteraciones += 1

    if mostrar_grafico:
        graficar_iteraciones(g, x_vals)

    return x_vals[-1], iteraciones + 1

def graficar_iteraciones(g, x_vals):
    x = np.linspace(min(x_vals) - 0.5, max(x_vals) + 0.5, 400)
    y = g(x)

    plt.figure(figsize=(8,6))
    plt.plot(x, y, label="g(x)", color='blue')
    plt.plot(x, x, label="y = x", linestyle='--', color='gray')
    
    # Dibujar la "escalera" de convergencia
    for i in range(len(x_vals)-1):
        # Línea vertical: (x_n, x_n) -> (x_n, x_{n+1})
        plt.plot([x_vals[i], x_vals[i]], [x_vals[i], x_vals[i+1]], color='green', linestyle='-')
        # Línea horizontal: (x_n, x_{n+1}) -> (x_{n+1}, x_{n+1})
        plt.plot([x_vals[i], x_vals[i+1]], [x_vals[i+1], x_vals[i+1]], color='green', linestyle='-')

    plt.scatter(x_vals, [g(x) for x in x_vals[:-1]] + [x_vals[-1]], color='red', zorder=5)
    plt.title("Método del Punto Fijo")
    plt.xlabel("x")
    plt.ylabel("g(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejemplo: Aproximar √3 usando g(x) = 0.5 * (x + 3 / x)
def g(x):
    return 5/x**2 + 2

# Ejecutar con x0 = 1 y tolerancia 1e-4
aprox, iters = punto_fijo(g, x0=1, tol=1e-3)
print(f"\nAproximación final: {aprox:.10f} en {iters} iteraciones.")