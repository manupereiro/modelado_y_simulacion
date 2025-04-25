# Ejemplo de uso
import math
import numpy as np

def aitken_punto_fijo_polynomial(g, x0, tol=1e-5, max_iter=100):
    """
    Método de aceleración de Aitken para la iteración de punto fijo en polinomios.
    
    g: Función de iteración asociada al polinomio.
    x0: Valor inicial.
    tol: Tolerancia para la convergencia.
    max_iter: Número máximo de iteraciones.
    """
    iteraciones = 0
    x_prev = x0
    while iteraciones < max_iter:
        # Realizar tres iteraciones de punto fijo
        x1 = g(x_prev)
        x2 = g(x1)
        
        # Aplicar la fórmula de Aitken
        numerador = (x1 - x_prev) ** 2
        denominador = x2 - 2 * x1 + x_prev
        if abs(denominador) < tol:  # Evitar división por cero
            break
        
        x_accelerated = x_prev - numerador / denominador
        
        # Verificar convergencia
        if abs(x_accelerated - x_prev) < tol:
            return x_accelerated, iteraciones
        
        x_prev = x_accelerated
        iteraciones += 1
    return x_prev, iteraciones

def punto_fijo(g, x0, tol=1e-5, max_iter=100):
    """
    Método de iteración de punto fijo.
    
    g: Función de iteración.
    x0: Valor inicial.
    tol: Tolerancia para la convergencia.
    max_iter: Número máximo de iteraciones.
    """
    iteraciones = 0
    x_prev = x0
    while iteraciones < max_iter:
        x_next = g(x_prev)
        if abs(x_next - x_prev) < tol:
            break
        x_prev = x_next
        iteraciones += 1
    return x_next, iteraciones
 
 
def aitken_punto_fijo(g, x0, tol=1e-5, max_iter=100):
    """
    Método de aceleración de Aitken para la iteración de punto fijo.
    
    g: Función de iteración.
    x0: Valor inicial.
    tol: Tolerancia para la convergencia.
    max_iter: Número máximo de iteraciones.
    """
    iteraciones = 0
    x_prev = x0
    while iteraciones < max_iter:
        # Realizar tres iteraciones de punto fijo
        x1 = g(x_prev)
        x2 = g(x1)
        
        # Aplicar la fórmula de Aitken
        numerador = (x1 - x_prev) ** 2
        denominador = x2 - 2 * x1 + x_prev
        if abs(denominador) < tol:  # Evitar división por cero
            break
        
        x_accelerated = x_prev - numerador / denominador
        
        # Verificar convergencia
        if abs(x_accelerated - x_prev) < tol:
            return x_accelerated, iteraciones
        
        x_prev = x_accelerated
        iteraciones += 1
    return x_prev, iteraciones
 
g = lambda x: (2/np.pi) + (4/x*np.pi)  # Función de iteración para el ejemplo
x0 = 1.4  # Valor inicial
 
# Iteración de punto fijo
resultado, iters = aitken_punto_fijo(g, x0)
resultado1, iters1 = punto_fijo(g, x0)
resultado2, iters2 = aitken_punto_fijo_polynomial(g, x0)
print(f"Resultado acelerado: {resultado}, Iteraciones: {iters}")
print(f"Resultado normal: {resultado1}, Iteraciones: {iters1}")
print(f"Resultado acelerado polinómico: {resultado2}, Iteraciones: {iters2}")