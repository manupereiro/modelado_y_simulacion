import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

n = 10000

def f(x):
    return (np.sin(x)/x)

def intervalo_de_confianza(media, error_estandar, alpha):
    # Calcular el valor crítico z para el nivel de confianza deseado
    z = stats.norm.ppf(1 - alpha / 2)
    intervalo_de_confianza = (media - z * error_estandar, media + z * error_estandar)
    return intervalo_de_confianza

def montecarlo1dimension(f,n):
    # Establecemos los limites del intervalo
    a,b = 0,np.pi

    # Calcular la longitud del intervalo
    longitud = b-a

    # Generar puntos aleatorios uniformemente distribuidos en el intervalo [a, b]
    datos = np.random.uniform(a,b,n)

    #Evaluo los puntos en la funcion
    evaluados = f(datos)

    #Calculos estadisticos sobre los puntos evaluados
    media = np.mean(evaluados) 
    integral_estimada = longitud * media
    desv_estandar = np.std(evaluados, ddof=1)
    error_estandar = desv_estandar/np.sqrt(n)
    varianza = np.var(evaluados, ddof=1)
    intervalo = intervalo_de_confianza(integral_estimada, error_estandar, 0.05)

    print(f"Integral de la funcion")
    print(f'Limite inferior: {a}')
    print(f'Limite superior: {b}')
    print(f"Longitud del intervalo: {longitud}")
    print(f"Media: {media}")
    print(f"Integral estimada: {integral_estimada}")
    print(f"Desviacion estandar: {desv_estandar}")
    print(f"Error estandar: {error_estandar}")
    print(f"Varianza: {varianza}")
    print(f"Intervalo de confianza: {intervalo}")

def montecarlo2dimenisones(f,n):
    # Definir los límites del rectángulo
    a,b = 0,1
    c,d = 1,2

    # Generar puntos aleatorios uniformemente distribuidos en el rectángulo
    datos_x = np.random.uniform(a, b, n)
    datos_y = np.random.uniform(c, d, n)
    
    # Calcular el área del dominio de integración
    area = (b - a) * (d - c)
    
    # Evaluar la función en los puntos generados
    evaluados = f(datos_x, datos_y)
    
    # Cálculos estadísticos
    media = np.mean(evaluados)
    integral_estimada = area * media
    desv_estandar = np.std(evaluados, ddof=1)
    error_estandar = desv_estandar / np.sqrt(n)
    varianza = np.var(evaluados, ddof=1)
    intervalo = intervalo_de_confianza(integral_estimada, error_estandar, 0.05)
    
    # Resultados
    print(f"Integral doble de la función")
    print(f'Límites en x: [{a}, {b}]')
    print(f'Límites en y: [{c}, {d}]')
    print(f"Área del dominio: {area}")
    print(f"Media de las evaluaciones: {media}")
    print(f"Integral estimada: {integral_estimada}")
    print(f"Desviación estándar: {desv_estandar}")
    print(f"Error estándar: {error_estandar}")
    print(f"Varianza: {varianza}")
    print(f"Intervalo de confianza:  {intervalo}")

#montecarlo1dimension(f,n)

montecarlo2dimenisones(f,n)