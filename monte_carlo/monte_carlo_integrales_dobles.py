import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)  # Para reproducibilidad

def intervalo_de_confianza(media, error_estandar, alpha):
    """Calcula el intervalo de confianza para la integral estimada."""
    z = stats.norm.ppf(1 - alpha / 2)
    intervalo_de_confianza = (media - z * error_estandar, media + z * error_estandar)
    return intervalo_de_confianza

def montecarlo_integral_doble(f, a, b, c, d, n=10000, alpha=0.05, visualizar=False):
    """
    Aproxima una integral doble usando el método de Monte Carlo.
    
    Parámetros:
    -----------
    f : callable
        Función de dos variables f(x,y) a integrar.
    a, b : float
        Límites inferior y superior para la variable x.
    c, d : float
        Límites inferior y superior para la variable y.
    n : int, opcional
        Número de puntos aleatorios a generar. Por defecto es 10000.
    alpha : float, opcional
        Nivel de significancia para el intervalo de confianza. Por defecto es 0.05 (95%).
    visualizar : bool, opcional
        Si es True, genera una visualización de los puntos. Por defecto es False.
        
    Retorna:
    --------
    float
        El valor estimado de la integral.
    """
    # Generar puntos aleatorios uniformemente distribuidos en el rectángulo [a,b]×[c,d]
    puntos_x = np.random.uniform(a, b, n)
    puntos_y = np.random.uniform(c, d, n)
    
    # Calcular el área del dominio de integración (rectángulo)
    area = (b - a) * (d - c)
    
    # Evaluar la función en los puntos generados
    evaluados = np.zeros(n)
    for i in range(n):
        evaluados[i] = f(puntos_x[i], puntos_y[i])
    
    # Cálculos estadísticos
    media = np.mean(evaluados)
    integral_estimada = area * media
    desv_estandar = np.std(evaluados, ddof=1)
    error_estandar = desv_estandar / np.sqrt(n)
    varianza = np.var(evaluados, ddof=1)
    intervalo = intervalo_de_confianza(integral_estimada, error_estandar, alpha)
    
    # Resultados
    print(f"Integral doble de la función")
    print(f'Límites en x: [{a}, {b}]')
    print(f'Límites en y: [{c}, {d}]')
    print(f"Área del dominio: {area}")
    print(f"Número de puntos: {n}")
    print(f"Media de las evaluaciones: {media:.6f}")
    print(f"Integral estimada: {integral_estimada:.6f}")
    print(f"Desviación estándar: {desv_estandar:.6f}")
    print(f"Error estándar: {error_estandar:.6f}")
    print(f"Varianza: {varianza:.6f}")
    print(f"Intervalo de confianza ({(1-alpha)*100}%): ({intervalo[0]:.6f}, {intervalo[1]:.6f})")

    # Visualización opcional
    if visualizar:
        fig = plt.figure(figsize=(12, 8))
        
        # Gráfico de dispersión 2D de los puntos coloreados por valor
        ax1 = fig.add_subplot(1, 2, 1)
        scatter = ax1.scatter(puntos_x, puntos_y, c=evaluados, cmap='viridis', 
                             alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax1, label='f(x,y)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Puntos aleatorios con valores de f(x,y)')
        ax1.grid(True)
        ax1.set_xlim(a, b)
        ax1.set_ylim(c, d)
        
        # Gráfico 3D de superficie (aproximada)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        # Crear una malla para aproximar la función
        x_grid = np.linspace(a, b, 30)
        y_grid = np.linspace(c, d, 30)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        
        # Evaluar la función en la malla
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(X[i, j], Y[i, j])
        
        surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                               edgecolor='none')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('f(x,y)')
        ax2.set_title('Superficie de la función')
        
        plt.tight_layout()
        plt.show()
    
    return integral_estimada

# Ejemplo de uso con una función de dos variables
def ejemplo_funcion(x, y):
    return np.exp(2*x-y)

# Para ejecutar el ejemplo:
if __name__ == "__main__":
    # Dimensiones del dominio de integración
    a, b = 0, 1  # límites de x
    c, d = 1, 2  # límites de y
    
    # Número de puntos
    n = 40000

    alpha = 0.05
    
    # Ejecutar la integración de Monte Carlo
    resultado = montecarlo_integral_doble(ejemplo_funcion, a, b, c, d, n=n, alpha=alpha, visualizar=True)
    
    # Para esta función, la integral analítica es ∫∫ x*y² dxdy = [0,1]×[1,2] = 7/6 ≈ 1.1667
    print(f"\nValor analítico exacto: {7/6:.6f}")
    print(f"Error absoluto: {abs(resultado - 7/6):.6f}")