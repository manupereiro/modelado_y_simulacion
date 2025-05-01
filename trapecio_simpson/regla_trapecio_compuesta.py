import numpy as np
import sympy as sp # Importar sympy para cálculo simbólico

# --- Funciones Modulares ---

def funcion(x):
    """Define la función a integrar (versión numérica)."""
    return np.sqrt(2)*np.exp(x**2)  # Puedes cambiarla por cualquier otra función

# --- Funciones Simbólicas (usando SymPy) ---
x_sym = sp.symbols('x') # Definir el símbolo x para sympy

def funcion_simbolica():
    """Define la función a integrar (versión simbólica)."""
    # Asegúrate de que esta expresión coincida con la función numérica
    return 6 + 3 * sp.cos(x_sym)  # Puedes cambiarla por otra función simbólica 

def segunda_derivada_exacta(f_sym):
    """Calcula la segunda derivada simbólica exacta."""
    return sp.diff(f_sym, x_sym, 2)

# --- Funciones Numéricas (usando las simbólicas) ---

def regla_trapecio_compuesta(f, a, b, n):
    """
    Calcula la integral aproximada usando la regla del trapecio compuesta.

    Args:
        f (callable): La función numérica a integrar.
        a (float): Límite inferior de integración.
        b (float): Límite superior de integración.
        n (int): Número de subintervalos.

    Returns:
        float: La integral aproximada.
    """
    if n <= 0:
        raise ValueError("El número de subintervalos 'n' debe ser positivo.")
    
    h = (b - a) / n
    x_puntos = np.linspace(a, b, n + 1)
    y = f(x_puntos) # Usa la función numérica aquí
    
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[-1])
    return integral

def calcular_error_trapecio(f_sym, a, b, n):
    """
    Estima el error de truncamiento para la regla del trapecio compuesta
    usando la segunda derivada exacta calculada con SymPy.

    Args:
        f_sym (sympy.Expr): La función simbólica (expresión de SymPy).
        a (float): Límite inferior de integración.
        b (float): Límite superior de integración.
        n (int): Número de subintervalos.

    Returns:
        float: Una estimación del error de truncamiento.
    """
    if n <= 0:
        raise ValueError("El número de subintervalos 'n' debe ser positivo.")

    # Calcular la segunda derivada exacta simbólicamente
    f_pp_sym = segunda_derivada_exacta(f_sym)
    
    # Convertir la segunda derivada simbólica a una función numérica evaluable
    f_pp_num = sp.lambdify(x_sym, f_pp_sym, 'numpy')

    # Encontrar el máximo absoluto de la segunda derivada exacta en el intervalo [a, b]
    # (Se muestrea en varios puntos para obtener una estimación precisa)
    puntos_muestra = np.linspace(a, b, 1000) # Más puntos para mayor precisión con la derivada exacta
    segundas_derivadas_valores = f_pp_num(puntos_muestra)
    
    # Manejar posibles NaNs o Infs si la derivada no está definida en algún punto (aunque para sin(x) no debería pasar)
    segundas_derivadas_valores = np.nan_to_num(segundas_derivadas_valores, nan=0.0, posinf=0.0, neginf=0.0) 
    
    max_f_pp_abs = np.max(np.abs(segundas_derivadas_valores)) 

    # Fórmula del error de truncamiento (usando el máximo absoluto de f'')
    error_estimado = -((b - a)**3) / (12 * n**2) * max_f_pp_abs 
    # Nota: La fórmula original usa f''(xi). Usar el máximo absoluto nos da 
    # una cota superior para el valor absoluto del error.
    
    return error_estimado

# --- Ejecución Principal ---

# Parámetros de integración
a = 0          # Límite inferior
b = 1  # Límite superior
n = 10         # Número de subintervalos (ajusta según la precisión deseada)

# Obtener la función simbólica
f_simbolica = funcion_simbolica()

# Calcular la integral (usando la función numérica)
integral_aproximada = regla_trapecio_compuesta(funcion, a, b, n)
print(f"Integral aproximada con la regla del trapecio compuesta: {integral_aproximada:.6f}")

# Calcular el error estimado (usando la función simbólica)
error_estimado = calcular_error_trapecio(f_simbolica, a, b, n)
print(f"Error de truncamiento estimado (usando derivada exacta): {error_estimado:.6f}")
print(f"Cota superior del error absoluto: {abs(error_estimado):.6f}")

# Valor exacto (para comparación, si se conoce)
valor_exacto = sp.integrate(f_simbolica, (x_sym, a, b)).evalf()
error_real = valor_exacto - integral_aproximada
print(f"Valor exacto de la integral: {valor_exacto:.6f}")
print(f"Error real: {error_real:.6f}")

# Opcional: Imprimir la segunda derivada simbólica encontrada
# f_pp_sym = segunda_derivada_exacta(f_simbolica)
# print(f"Segunda derivada simbólica: {f_pp_sym}")