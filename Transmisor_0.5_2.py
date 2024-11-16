import numpy as np
from scipy.integrate import quad
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from scipy.signal import periodogram, upfirdn

from multiprocessing import cpu_count
from tqdm import tqdm
import time
from PIL import Image
import matplotlib.pyplot as plt
from scipy.special import erfc, erfcinv
start_time = time.time()

#-------------------TRANSMISOR-----------------------#

# Leer la imagen
image = Image.open('imagen_linda.jpg')

# Convertir la imagen a un array de numpy
image_array = np.array(image)

# Verificar las dimensiones de la imagen
print(f"Dimensiones de la imagen: {image_array.shape}")

# Descomponer en las tres matrices de colores
red_channel = image_array[:, :, 0]
green_channel = image_array[:, :, 1]
blue_channel = image_array[:, :, 2]

# Asegurarse de que las dimensiones son correctas
assert red_channel.shape == (108, 192)
assert green_channel.shape == (108, 192)
assert blue_channel.shape == (108, 192)

# Convertir cada canal de color (0-255) directamente a bits (sin bucle)
red_bits = np.unpackbits(red_channel.flatten())
green_bits = np.unpackbits(green_channel.flatten())
blue_bits = np.unpackbits(blue_channel.flatten())

# Concatenar los arrays de bits en un solo vector 1D
concatenated_bits = np.concatenate((red_bits, green_bits, blue_bits))

# Ejemplo: Imprimir los primeros 80 bits para verificar
print("Concatenated bits (first 80 bits):", concatenated_bits[:80])

data = concatenated_bits.flatten()


np.savetxt(f'data_original.csv', data, delimiter=',')

#-------------------Modulacion-----------------------#
"""
# Generar datos
largo_data = len(data)  # Ajustar el tamaño de datos según sea necesario 10%=497664
A = 1
f_c = 100  # Frecuencia máxima de un Arduino Uno para generar funciones cuadradas
omega_c = 2 * np.pi * f_c
T_muestreo = 1e-5 # Periodo mínimo de parpadeo de un led blanco
interval_step = T_muestreo
f0 = 100  # Frecuencia máxima de un Arduino Uno para generar funciones cuadradas
omega = 2 * np.pi * f0

# Crear el vector de tiempo
t = np.arange(0, largo_data * T_muestreo, T_muestreo)
print(t)

# Calcular la función cos(omega t)
cos_omega_t = A * np.cos(omega * t)

# Multiplicar el array de números aleatorios por la función cos(omega t)
result_array = data * cos_omega_t

total_duration = largo_data * T_muestreo

# Crear el vector de tiempo desde 0 hasta total_duration con incrementos de interval_step
time_vector = np.arange(0, total_duration, interval_step)

# Precalcular los valores de cos(omega * t) para ahorrar cálculos repetidos
cos_omega_time_vector = A * np.cos(omega * time_vector)

# Definir la función f(t)
def f(index, data):
    if data[index] == 0:
        return 0
    else:
        return cos_omega_time_vector[index]
    
    
# Duración de cada bit en muestras
samples_per_bit = len(time_vector) // largo_data

# Preasignar el vector de modulación para calzar con el time_vector
modulation = np.repeat(data, samples_per_bit)

# Si el tamaño de time_vector no es múltiplo exacto de largo_data, recortar o ajustar modulation
modulation = modulation[:len(time_vector)]  # Aseguramos que tenga el mismo tamaño que time_vector

# Señal portadora
carrier = A * np.cos(omega * time_vector)

# Señal modulada en amplitud: data binaria * cos(w_c t)
modulated_signal = modulation * carrier

# Graficar la señal modulada
plt.figure(figsize=(10, 4))
plt.plot(time_vector, modulated_signal, color='k')
plt.title(r'$data(t) \cdot \cos(\omega_c t)$ - Modulación OOK con datos binarios')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()
"""
#---------------------Modulación----------------------#

# Parámetros
A = 1  # Amplitud de la portadora
f_c = 100  # Frecuencia de la señal portadora (Hz)
omega_c = 2 * np.pi * f_c
largo_data = len(data)  # Número de bits en la data
samples_per_bit = 20  # Número de muestras por cada bit


# Calcular el total de muestras necesarias
total_samples = largo_data * samples_per_bit

# Duración total basada en el número total de muestras y la frecuencia de la señal portadora
T_muestreo = 1 / (f_c * samples_per_bit)  # Asegurando 10 muestras por período de la señal portadora
total_duration = T_muestreo * total_samples

# Crear el vector de tiempo
time_vector = np.linspace(0, total_duration, total_samples, endpoint=False)

# Expandir los datos para que cada bit ocupe 10 muestras
modulation = np.repeat(data, samples_per_bit)

# Señal portadora ajustada al vector de tiempo
carrier = A * np.sin(omega_c * time_vector)

# Señal modulada en amplitud: data binaria * sin(omega_c * t)
modulated_signal = modulation * carrier

# Graficar la señal modulada
plt.figure(figsize=(10, 6))
plt.plot(time_vector, modulated_signal, color='r')
plt.title(r'Modulación OOK con 20 muestras por bit')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()


# --- Calcular la potencia de la señal modulada ---
signal_power = np.mean(modulated_signal ** 2)

#-----------------------------Datos--------------------------------------------#


#Ruidos del sistema OCC

i_d = 202e-6  # Corriente de oscuridad en Amperios (C/s) 1.0 * q
# Cálculo de B
T_delay_row=18.904e-6
B = 1 / T_delay_row # Ancho de banda en Hz

q = 1.6e-19  # Carga del electrón

# Cálculo de delta_sh^2
delta_sh_squared = 2 * q * i_d * B

# Mostrar el resultado
print(f"El valor de delta_sh^2 es: {delta_sh_squared:.4e} A^2")

# Constantes
K_b = 1.38e-23  # Constante de Boltzmann en J/K
T_n = 300
G_v= 3.5  # Temperatura en Kelvin (ejemplo a temperatura ambiente, 27°C)
G_iso = 200  # Ganancia isotrópica
T_n = 300  # Temperatura efectiva en Kelvin (ejemplo a temperatura ambiente, 27°C). SUPOSIC

# Cálculo de B
B = 1 / T_delay_row  # Ancho de banda en Hz

# Cálculo de delta_th^2
delta_th_squared = (4 * K_b * T_n * B) / G_iso
# Valor de q calculado anteriormente
qq = 1 / 8  # en voltios

# Cálculo de sigma_adc^2
sigma_adc_squared = (qq ** 2) / 12

q = 1.6e-19  # Carga del electrón
Ib = 202e-6  # Corriente de ruido de fondo + interferencia
N0 = 2 * q * Ib + sigma_adc_squared + delta_th_squared # Densidad espectral de todos los ruidos
R = 1  # Responsividad del fotodetector
Rb = 1e3  # Tasa de bits
Tb = 1 / Rb  # Duración de un bit
sig_length = len(data)  # Número de bits
nsamp = 20  # Muestras por símbolo
Tsamp = Tb / nsamp  # Tiempo de muestreo
EbN0_dB = np.arange(-5,17,0.25)  # Relación Eb/N0 en dB desde -5 hasta 16 con steps de 0.25
SNR = 10 ** (EbN0_dB / 10)  # Relación SNR en escala lineal
randint=np.random.randn((4976640*2))

# Rango de SNR en dB
snr_range_dB = np.arange(-5, 17, 0.25)  # De -5 dB a 16 dB con steps de 0.25


for SNR_dB in snr_range_dB:
    # Calcular SNR en escala lineal
    SNR_linear = 10 ** (SNR_dB / 10)
    
    # Calcular la potencia promedio P_avg en función del SNR
    P_avg = np.sqrt(N0 * Rb * SNR_linear / (2 * R ** 2))  # Potencia óptica promedio
    i_peak = 2 * R * P_avg  # Amplitud pico de la señal
    Ep = i_peak ** 2 * Tb  # Energía pico por bit
    
    # Señal transmitida con potencia óptica y responsividad aplicada
    Tx_signal = modulated_signal * i_peak * R
    
    # Supongamos que tienes la señal transmitida `Tx_signal` y el valor de SNR en dB `SNR_dB`
    Tx_signal = np.array(Tx_signal)  # Convertir la lista de Tx_signal a un array de numpy si es que no lo es ya
    
    # Redimensionar a (497664, 10)
    Tx_signal = Tx_signal.reshape(497664, 20)    
    
    # Nombre del archivo basado en el valor de SNR en dB
    filename = f'Tx_signal_{SNR_dB}dB.csv'
    
    # Guardar `Tx_signal` en un archivo CSV, cada valor en una fila
    np.savetxt(filename, Tx_signal, delimiter=',')    

end_time = time.time()
execution_time = end_time - start_time
print(f"Tiempo de ejecución total: {execution_time} segundos")