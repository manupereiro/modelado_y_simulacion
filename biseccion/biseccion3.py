import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Método de Bisección original (sin modificar)
def biseccion(f, a, b, iteraciones=100, tolerancia=1e-6, precision=5, mostrar_tabla=True):
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe tener signos opuestos en los extremos del intervalo [a, b].")
    
    results = []
    for i in range(iteraciones):
        c = round((a + b) / 2.0, precision)
        fc = round(f(c), precision)
        results.append([i+1, round(a, precision), round(b, precision), c, fc])
        
        if abs(fc) < tolerancia or (b - a) / 2.0 < tolerancia:
            if mostrar_tabla:
                print(tabulate(results, headers=["Iteración", "a", "b", "c", "f(c)"], tablefmt="grid"))
            return c
        
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    
    raise ValueError("El método no convergió o faltan iteraciones.")

# Función para encontrar cambios de signo (posibles raíces)
def encontrar_cambios_signo(f, inicio, fin, num_puntos=1000):
    """
    Encuentra intervalos donde la función cambia de signo (posibles raíces).
    """
    x_vals = np.linspace(inicio, fin, num_puntos)
    y_vals = [f(x) for x in x_vals]
    
    intervalos = []
    for i in range(len(y_vals) - 1):
        if y_vals[i] * y_vals[i+1] <= 0:  # Se detectó un cambio de signo
            intervalos.append((x_vals[i], x_vals[i+1]))
    
    return intervalos

# Función para encontrar todas las raíces
def encontrar_todas_raices(f, inicio, fin, iteraciones=100, tolerancia=1e-6, precision=5):
    """
    Encuentra todas las raíces de una función en un rango dado.
    """
    # Encontrar intervalos donde hay cambios de signo
    intervalos = encontrar_cambios_signo(f, inicio, fin)
    
    # Aplicar método de bisección a cada intervalo
    raices = []
    for i, (a, b) in enumerate(intervalos):
        try:
            print(f"\nBuscando raíz #{i+1} en el intervalo [{a:.4f}, {b:.4f}]:")
            root = biseccion(f, a, b, iteraciones, tolerancia, precision)
            # Verificar si esta raíz ya está en la lista (evitar duplicados)
            if not any(abs(root - r) < tolerancia for r in raices):
                raices.append(root)
        except ValueError as e:
            # Si bisección falla, saltamos este intervalo
            print(f"Saltando intervalo [{a:.4f}, {b:.4f}]: {e}")
    
    return raices

# Función para graficar todas las raíces
def graficar_todas_raices(f, inicio, fin, raices, precision):
    """
    Grafica la función y todas sus raíces.
    """
    # Graficar la función
    x = np.linspace(inicio, fin, 1000)
    y = f(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='$f(x)$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    
    # Marcar todas las raíces encontradas
    for i, raiz in enumerate(raices):
        plt.plot(raiz, f(raiz), 'ro')
        plt.annotate(f'$r_{i+1} = {raiz:.{precision}f}$', 
                     (raiz, f(raiz)), 
                     textcoords="offset points",
                     xytext=(0,10), 
                     ha='center')
    
    # Añadir leyenda y mostrar la gráfica
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Gráfica del polinomio y sus {len(raices)} raíces')
    plt.show()

# ===================== CONFIGURACIÓN =====================
# Definir el polinomio para el cual quieres encontrar las raíces
def f(x):
    # Ejemplo: x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    return (x+2)*(x+1)*((x-1)**3)*(x-2)

# Rango de búsqueda de raíces
inicio = -1.5
fin = 1.75

# Parámetros que podés cambiar directamente
tolerancia = 1e-3
precision = 10
# ========================================================

# Encontrar todas las raíces utilizando el método modificado
raices = encontrar_todas_raices(f, inicio, fin, tolerancia=tolerancia, precision=precision)

# Ordenar las raíces de menor a mayor
raices.sort()

# Imprimir las raíces encontradas
print("\nRaíces encontradas:")
for i, raiz in enumerate(raices):
    print(f"Raíz {i+1}: {raiz:.{precision}f}")

# Graficar la función y todas sus raíces
graficar_todas_raices(f, inicio, fin, raices, precision)