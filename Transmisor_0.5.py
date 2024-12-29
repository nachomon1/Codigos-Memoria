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

def ejecutar_transmisor(amplitud):
    

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
    A = amplitud  # Amplitud de la portadora
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
    #plt.show()
    
    
    # Parámetros iniciales
    power_vector_final=np.zeros(len(data))
    for i in range(0,len(data)):
        if data[i] == 0:
            power_vector_final[i]= 0
        else:
            power_vector_final[i] = (amplitud**2)/2
    
    print(power_vector_final[0:40])
    
    """
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
    
    # Inicialización de variables
    P_avg = np.zeros(len(SNR))  # Potencia óptica transmitida promedio
    i_peak = np.zeros(len(SNR))  # Amplitud pico de la señal
    Ep = np.zeros(len(SNR))  # Energía pico por bit
    sgma = np.zeros(len(SNR))  # Varianza del ruido
    ber = np.zeros(len(SNR))  # Tasa de error de bits (BER)
    
    # Simulación de probabilidad de errores
    for i in range(len(SNR)):
        P_avg[i] = np.sqrt(N0 * Rb * SNR[i] / (2 * R**2))
        i_peak[i] = 2 * R * P_avg[i]
        Ep[i] = i_peak[i]**2 * Tb
        sgma[i] = np.sqrt(N0 / 2 / Tsamp)
        
        # Generar la señal aleatoria OOK
        OOK = data  # 'data' es la señal binaria original
    
        # Filtrado transmisor y receptor
        pt = np.ones(nsamp) * i_peak[i]
        rt = pt
        
        # Pulso de la señal transmitida
        Tx_signal = np.repeat(OOK, nsamp) * i_peak[i]  # Repetir cada bit nsamp veces
        
        # Señal recibida con ruido AWGN
        Rx_signal = R * Tx_signal + sgma[i] * randint
        
        # Salida del filtro adaptado
        MF_out = np.convolve(Rx_signal, rt) * Tsamp
        
        # Muestreo al final del período del bit
        MF_out_downsamp = MF_out[nsamp - 1::nsamp][:sig_length]  # Obtener una muestra por bit
    
        # Umbral de decisión
        Rx_th = np.zeros(sig_length)  # Rx_th debe tener la misma longitud que MF_out_downsamp
        Rx_th[MF_out_downsamp > Ep[i] / 2 ] = 1  # Umbral de decisión para la señal recibida
        
        # Cálculo de error de bits (comparar OOK original con las decisiones)
        nerr = np.sum(OOK != Rx_th)
        ber[i] = nerr / sig_length  # Calcular BER para este valor de SNR
    
    # Gráfico de la curva de probabilidad de error de bits (BER)
    plt.figure()
    plt.semilogy(EbN0_dB, ber, 'b', label='simulation')
    plt.semilogy(EbN0_dB, erfc(np.sqrt(10 ** (EbN0_dB / 10))) / 2, 'r-X', linewidth=2, label='theory')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Eb/No, dB')
    plt.ylabel('Bit Error Rate')
    plt.title('Bit error probability curve for OOK modulation')
    plt.ylim([1e-12, 1])
    plt.show()
    
    #
    
    # Parámetros
    p_avg = 0.5  # Potencia óptica promedio
    R = 1  # Sensibilidad del fotodetector
    
    df = Rb / 100  # Resolución espectral
    
    # Vector de frecuencias
    f = np.arange(0, 5 * Rb + df, df)
    x = f * Tb  # Frecuencia normalizada
    
    # Cálculos de PSD
    temp1 = np.sinc(x) ** 2  # sinc(x) es sin(pi*x) / (pi*x)
    a = 2 * R * p_avg  # Potencia máxima es dos veces la potencia promedio
    p = (a ** 2 * Tb) * temp1  # PSD
    
    # Normalización de potencia por energía por bit
    p = p / ((p_avg * R) ** 2 * Tb)
    
    # Visualización de PSD
    plt.plot(f, p)
    plt.title('PSD Analítica de OOK-NRZ')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Densidad Espectral de Potencia (PSD)')
    plt.grid(True)
    plt.show()
    
    
    
    # --- Agregar ruido AWGN ---
    
    # Calcular la potencia de la señal modulada
    signal_power = np.mean(modulated_signal ** 2)
    """

    R=1
    # Señal recibida (señal modulada)
    received_signal = R * modulated_signal
    
    received_signal_final=received_signal.reshape(497664, 20)
     
    # Guardar los vectores en archivos CSV
    np.savetxt('data.csv', data, delimiter=',')
    np.savetxt(f'received_signal_{amplitud}.csv', received_signal_final, delimiter=',')
    np.savetxt(f'power_vector_{amplitud}.csv', power_vector_final, delimiter=',')    
   
    
    """
    #--------------Potencia----------------------#
    
    # --- Optimización de la función para calcular la potencia de un bit ---
    def calcular_potencia_por_bit(index):
        start = index * samples_per_bit
        end = start + samples_per_bit
        # Calcular la potencia usando np.mean de la señal recibida (incluyendo el ruido)
        return np.mean(received_signal[start:end] ** 2)
    
    # --- Paralelizar el cálculo de la potencia de los bits ---
    def calcular_potencia_wrapper(index):
        return calcular_potencia_por_bit(index)
    
    # Preparar los índices para el cálculo paralelo (esto solo debe tener tantos índices como bits, no como muestras)
    indices = np.arange(len(data))  # Usamos largo_data en lugar de len(time_vector)
    
    # Paralelizar el cálculo de la potencia
    if __name__ == '__main__':
        cpu_count = multiprocessing.cpu_count()
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            power_vector = list(tqdm(executor.map(calcular_potencia_wrapper, indices), total=len(indices)))
        
        # Convertir la lista de potencias a un array de numpy para facilitar su uso posterior
        power_vector = np.array(power_vector)
    
        # Imprimir los primeros 20 resultados
        print(f"Vector de data: {data[:20]}")
        print(f"Vector de potencia P (con AWGN): {power_vector[:20]}")
        print(f"Vector de data modulada (con ruido): {received_signal[:20]}")
    
        print(f"Tamaño de time_vector: {len(time_vector)}")
        print(f"Tamaño de power_vector: {len(power_vector)}")
        print(f"Tamaño de data: {len(data)}")
        print(f"Tamaño de data modulada: {len(modulated_signal)}")
    
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución: {execution_time} segundos")

    # Rango de SNR en dB
    # Rango de SNR en dB
    snr_range_dB = np.arange(-5, 17, 0.25)  # De -5 dB a 16 dB con steps de 0.25
    
    for SNR_dB in snr_range_dB:
        # Calcular SNR en escala lineal
        SNR_linear = 10 ** (SNR_dB / 10)
        
        # Calcular la potencia del ruido en función del SNR
        noise_power = signal_power / SNR_linear
        
        # Generar el ruido AWGN con la potencia calculada
        noise = np.sqrt(noise_power) * np.random.randn(len(modulated_signal))
        
        # Señal recibida (señal modulada + ruido AWGN)
        received_signal = modulated_signal + noise
    
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
        indices = np.arange(len(data))
    
        # Paralelizar el cálculo de la potencia
        cpu_count = multiprocessing.cpu_count()
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            power_vector = list(tqdm(executor.map(calcular_potencia_wrapper, indices), total=len(indices)))
        
        # Convertir la lista de potencias a un array de numpy para facilitar su uso posterior
        power_vector = np.array(power_vector)
        
        # Guardar el power_vector en un archivo CSV con una columna y 497664 filas
        filename = f'power_vector_{SNR_dB}dB.csv'
        np.savetxt(filename, power_vector, delimiter=',')
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución total: {execution_time} segundos")
    """
    
"""
ejecutar_transmisor(620)    
ejecutar_transmisor(640)
ejecutar_transmisor(660)
ejecutar_transmisor(680)
ejecutar_transmisor(720)
ejecutar_transmisor(740)
ejecutar_transmisor(760)
ejecutar_transmisor(780)
ejecutar_transmisor(820)
ejecutar_transmisor(840)
ejecutar_transmisor(860)
ejecutar_transmisor(880)
ejecutar_transmisor(920)
ejecutar_transmisor(940)
ejecutar_transmisor(960)
ejecutar_transmisor(980)
"""
ejecutar_transmisor(1020)
ejecutar_transmisor(1040)
ejecutar_transmisor(1060)
ejecutar_transmisor(1080)
ejecutar_transmisor(1100)


