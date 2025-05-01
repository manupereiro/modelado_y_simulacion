import math
import numpy as np
from tabulate import tabulate
import pandas as pd  # Importamos pandas

def g(x):
    """
    Función iteradora. 
    Cambia esta definición a la que uses en tu ejercicio.
    Por ejemplo: g(x) = (2/pi) + (4/(x*pi))
    """
    return np.sqrt((np.e)**x/3)

def aitken_acceleration(g, x0, tol=1e-3, max_iter=50, precision=10):
    """
    Aplica el método iterativo x_{n+1} = g(x_n) y acelera con Aitken.
    
    Parámetros:
    -----------
    - g: función iteradora
    - x0: aproximación inicial
    - tol: tolerancia para detenernos
    - max_iter: número máximo de iteraciones
    - precision: cantidad de dígitos a mostrar en la tabla
    
    Retorna:
    --------
    - tabla de iteraciones (lista de listas)
    - x_star final (último valor acelerado de Aitken que se compute)
    """
    
    tabla = []
    x_old = x0
    
    for i in range(1, max_iter+1):
        # Calculamos x1, x2
        x1 = g(x_old)
        x2 = g(x1)
        
        # Aitken: x0^* = x0 - ( (x1 - x0)^2 ) / ( x2 - 2x1 + x0 )
        denom = x2 - 2*x1 + x_old
        if abs(denom) < 1e-15:
            # Evitamos división por cero
            x_star = float('nan')
        else:
            x_star = x_old - ( (x1 - x_old)**2 ) / denom
        
        # Definimos el error como la diferencia entre x_old^* y x_old
        error = (abs(x_star - x2)/x_star)*100 if not math.isnan(x_star) else float('inf')
        
        # Guardamos en la tabla
        fila = [
            i,
            round(x_old, precision),
            round(x1,   precision),
            round(x2,   precision),
            round(x_star, precision) if not math.isnan(x_star) else "NaN",
            round(error, precision) if error != float('inf') else "NaN"
        ]
        tabla.append(fila)
        
        # Criterio de parada: si x_star es válido y el error < tol
        if error < tol:
            break
        
        x_old = x_star if not math.isnan(x_star) else x1  # avanzamos usando la versión acelerada (si no es NaN)
    
    return tabla, x_star

if __name__ == "__main__":
    # Parámetros de usuario
    x0_inicial = 0.5       # ejemplo de valor inicial
    tolerancia = 1e-4      # ejemplo de tolerancia
    num_iter_max = 20      # máximo de iteraciones
    digitos = 6            # precisión en la tabla
    
    # Ejecutar la aceleración de Aitken
    tabla_resultados, x_ultima = aitken_acceleration(
        g, 
        x0_inicial, 
        tol=tolerancia, 
        max_iter=num_iter_max, 
        precision=digitos
    )
    
    # Convertir a DataFrame de pandas y mostrar
    headers = ["i", "x0", "x1", "x2", "x0*", "error"]
    df_resultados = pd.DataFrame(tabla_resultados, columns=headers)
    df_resultados.set_index("i", inplace=True)
    print("\n--- Tabla de Iteraciones (Aitken) ---")
    print(df_resultados)  # Imprime directamente el DataFrame
    
    print(f"\nAproximación final tras Aitken: {x_ultima:.{digitos}f}")
    print(f"Valor real: {2/np.pi:.{digitos}f}")
    print(f"Error absoluto: {abs(x_ultima - 2/np.pi):.{digitos}f}")
    print(f"Error relativo: {abs((x_ultima - 2/np.pi) / (2/np.pi)):.{digitos}f}")
    print(f"Iteraciones realizadas: {len(tabla_resultados)}")