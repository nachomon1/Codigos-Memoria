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

import re


############-------Receptor1-------------#########
samples_per_bit=20

data_original = np.loadtxt('data.csv',delimiter=',')
print(data_original)
def ejecutar_receptor(nombre_archivo_csv):
    # Leer el archivo CSV y convertirlo en un array de NumPy
    Tx_signal = np.loadtxt(nombre_archivo_csv, delimiter=',')
    
    # Aplanar el array a 1D
    Tx_signal = Tx_signal.reshape(-1)    
    
    # Utilizamos una expresión regular para extraer el número decimal que está antes de 'dB'
    match = re.search(r'Tx_signal_([-\d\.]+)dB', nombre_archivo_csv)
    
    if match:
        SNR_dB = float(match.group(1))  # Convertir el valor extraído a float
        print(f"El valor de SNR es: {SNR_dB} dB")
    else:
        print("No se pudo encontrar el valor de SNR en el nombre del archivo.")
        return  # Salir de la función si no se encuentra el valor de SNR

    # Convertir el SNR de dB a lineal
    SNR_linear = 10**(SNR_dB / 10)

    # Supongamos que tienes la potencia de la señal original (puedes calcularla o asignarla)
    signal_power = np.mean(Tx_signal**2)

    # Calcular la potencia del ruido en función del SNR
    noise_power = signal_power / SNR_linear

    # Generar el ruido AWGN con la potencia calculada
    noise = np.sqrt(noise_power) * np.random.randn(len(Tx_signal))

    # Señal recibida (señal modulada con potencia + ruido AWGN)
    received_signal = Tx_signal + noise
    
    
    # --- Optimización de la función para calcular la potencia de un bit ---
    def calcular_potencia_por_bit(index):
        start = index * samples_per_bit
        end = start + samples_per_bit
        # Calcular la potencia usando np.mean de la señal recibida (incluyendo el ruido)
        return np.mean(received_signal[start:end] ** 2)
    
    # --- Paralelizar el cálculo de la potencia de los bits ---
    def calcular_potencia_wrapper(index):
        return calcular_potencia_por_bit(index)
    
    # Preparar los índices para el cálculo paralelo
    indices = np.arange(len(data_original))
    
    # Paralelizar el cálculo de la potencia
    cpu_count = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        power_vector = list(tqdm(executor.map(calcular_potencia_wrapper, indices), total=len(indices)))
    
    # Convertir la lista de potencias a un array de numpy para facilitar su uso posterior
    power_vector = np.array(power_vector)
    
    # Guardar el power_vector en un archivo CSV con una columna y 497664 filas
    filename = f'power_vector_{SNR_dB}dB.csv'
    np.savetxt(filename, power_vector, delimiter=',')


ejecutar_receptor('Tx_signal_-5.0dB.csv')
ejecutar_receptor('Tx_signal_-4.75dB.csv')
ejecutar_receptor('Tx_signal_-4.5dB.csv')
ejecutar_receptor('Tx_signal_-4.25dB.csv')
ejecutar_receptor('Tx_signal_-4.0dB.csv')
ejecutar_receptor('Tx_signal_-3.75dB.csv')
ejecutar_receptor('Tx_signal_-3.5dB.csv')
ejecutar_receptor('Tx_signal_-3.25dB.csv')
ejecutar_receptor('Tx_signal_-3.0dB.csv')
ejecutar_receptor('Tx_signal_-2.75dB.csv')
ejecutar_receptor('Tx_signal_-2.5dB.csv')
ejecutar_receptor('Tx_signal_-2.25dB.csv')
ejecutar_receptor('Tx_signal_-2.0dB.csv')
ejecutar_receptor('Tx_signal_-1.75dB.csv')
ejecutar_receptor('Tx_signal_-1.5dB.csv')
ejecutar_receptor('Tx_signal_-1.25dB.csv')
ejecutar_receptor('Tx_signal_-1.0dB.csv')
ejecutar_receptor('Tx_signal_-0.75dB.csv')
ejecutar_receptor('Tx_signal_-0.5dB.csv')
ejecutar_receptor('Tx_signal_-0.25dB.csv')
ejecutar_receptor('Tx_signal_0.0dB.csv')
ejecutar_receptor('Tx_signal_0.25dB.csv')
ejecutar_receptor('Tx_signal_0.5dB.csv')
ejecutar_receptor('Tx_signal_0.75dB.csv')
ejecutar_receptor('Tx_signal_1.0dB.csv')
ejecutar_receptor('Tx_signal_1.25dB.csv')
ejecutar_receptor('Tx_signal_1.5dB.csv')
ejecutar_receptor('Tx_signal_1.75dB.csv')
ejecutar_receptor('Tx_signal_2.0dB.csv')
ejecutar_receptor('Tx_signal_2.25dB.csv')
ejecutar_receptor('Tx_signal_2.5dB.csv')
ejecutar_receptor('Tx_signal_2.75dB.csv')
ejecutar_receptor('Tx_signal_3.0dB.csv')
ejecutar_receptor('Tx_signal_3.25dB.csv')
ejecutar_receptor('Tx_signal_3.5dB.csv')
ejecutar_receptor('Tx_signal_3.75dB.csv')
ejecutar_receptor('Tx_signal_4.0dB.csv')
ejecutar_receptor('Tx_signal_4.25dB.csv')
ejecutar_receptor('Tx_signal_4.5dB.csv')
ejecutar_receptor('Tx_signal_4.75dB.csv')
ejecutar_receptor('Tx_signal_5.0dB.csv')
ejecutar_receptor('Tx_signal_5.25dB.csv')
ejecutar_receptor('Tx_signal_5.5dB.csv')
ejecutar_receptor('Tx_signal_5.75dB.csv')
ejecutar_receptor('Tx_signal_6.0dB.csv')
ejecutar_receptor('Tx_signal_6.25dB.csv')
ejecutar_receptor('Tx_signal_6.5dB.csv')
ejecutar_receptor('Tx_signal_6.75dB.csv')
ejecutar_receptor('Tx_signal_7.0dB.csv')
ejecutar_receptor('Tx_signal_7.25dB.csv')
ejecutar_receptor('Tx_signal_7.5dB.csv')
ejecutar_receptor('Tx_signal_7.75dB.csv')
ejecutar_receptor('Tx_signal_8.0dB.csv')
ejecutar_receptor('Tx_signal_8.25dB.csv')
ejecutar_receptor('Tx_signal_8.5dB.csv')
ejecutar_receptor('Tx_signal_8.75dB.csv')
ejecutar_receptor('Tx_signal_9.0dB.csv')
ejecutar_receptor('Tx_signal_9.25dB.csv')
ejecutar_receptor('Tx_signal_9.5dB.csv')
ejecutar_receptor('Tx_signal_9.75dB.csv')
ejecutar_receptor('Tx_signal_10.0dB.csv')
ejecutar_receptor('Tx_signal_10.25dB.csv')
ejecutar_receptor('Tx_signal_10.5dB.csv')
ejecutar_receptor('Tx_signal_10.75dB.csv')
ejecutar_receptor('Tx_signal_11.0dB.csv')
ejecutar_receptor('Tx_signal_11.25dB.csv')
ejecutar_receptor('Tx_signal_11.5dB.csv')
ejecutar_receptor('Tx_signal_11.75dB.csv')
ejecutar_receptor('Tx_signal_12.0dB.csv')
ejecutar_receptor('Tx_signal_12.25dB.csv')
ejecutar_receptor('Tx_signal_12.5dB.csv')
ejecutar_receptor('Tx_signal_12.75dB.csv')
ejecutar_receptor('Tx_signal_13.0dB.csv')
ejecutar_receptor('Tx_signal_13.25dB.csv')
ejecutar_receptor('Tx_signal_13.5dB.csv')
ejecutar_receptor('Tx_signal_13.75dB.csv')
ejecutar_receptor('Tx_signal_14.0dB.csv')
ejecutar_receptor('Tx_signal_14.25dB.csv')
ejecutar_receptor('Tx_signal_14.5dB.csv')
ejecutar_receptor('Tx_signal_14.75dB.csv')
ejecutar_receptor('Tx_signal_15.0dB.csv')
ejecutar_receptor('Tx_signal_15.25dB.csv')
ejecutar_receptor('Tx_signal_15.5dB.csv')
ejecutar_receptor('Tx_signal_15.75dB.csv')
ejecutar_receptor('Tx_signal_16.0dB.csv')
ejecutar_receptor('Tx_signal_16.25dB.csv')
ejecutar_receptor('Tx_signal_16.5dB.csv')
ejecutar_receptor('Tx_signal_16.75dB.csv')
