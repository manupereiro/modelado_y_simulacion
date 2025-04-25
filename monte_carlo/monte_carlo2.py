import numpy as np
import scipy.stats as stats
 
def monte_carlo_double_exp(a1, b1, a2, b2, N=50000, confidence=0.90):  # Cambiado a 0.90
    # Generar muestras uniformes
    x_samples = np.random.uniform(a1, b1, N)
    y_samples = np.random.uniform(a2, b2, N)
 
    # Evaluar la función
    f_samples = np.exp(x_samples + y_samples)
 
    # Área del dominio
    area = (b1 - a1) * (b2 - a2)
 
    # Estimación de Monte Carlo
    estimate = area * np.mean(f_samples)
 
    # Desviación estándar del estimador
    std_f = np.std(f_samples, ddof=1)
    std_dev = (area * std_f) / np.sqrt(N)
 
    # Intervalo de confianza (90%)
    z_score = stats.norm.ppf((1 + confidence) / 2)  # Percentil 95% para alpha=0.10
    margin_of_error = z_score * std_dev
    conf_interval = (estimate - margin_of_error, estimate + margin_of_error)
 
    return estimate, conf_interval, std_dev
 
# Ejemplo de uso
a1, b1 = 0, 2
a2, b2 = 1, 3
N = 50
confidence = 0.90  # 90% de confianza
 
estimate, conf_interval, std_dev = monte_carlo_double_exp(a1, b1, a2, b2, N, confidence)
 
print(f"Estimación de la integral: {estimate:.6f}")
print(f"Intervalo de confianza al {confidence*100:.0f}%: ({conf_interval[0]:.6f}, {conf_interval[1]:.6f})")
print(f"Desviación estándar del estimador: {std_dev:.6f}")
 
# Valor teórico
theoretical_value = (np.exp(b1) - np.exp(a1)) * (np.exp(b2) - np.exp(a2))
print(f"Valor teórico: {theoretical_value:.6f}")