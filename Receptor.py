import numpy as np
import math
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from PIL import Image
from scipy.special import erfc, erfcinv
from mpl_toolkits.mplot3d import Axes3D  # Necesario para gráficos 3D

resultados_BER=[]
def ejecutar_receptor(nombre_archivo_csv):
    data_original = np.loadtxt('data.csv',delimiter=',')
    print(data_original)
    
    # Leer el archivo CSV y convertirlo en un array de NumPy
    P_tx = np.loadtxt(nombre_archivo_csv, delimiter=',')
    
    # Verificar el contenido del array
    print('Potencia transmitida recibida en el receptor, que tiene RGB:',  P_tx)
    
    #----------------Dividir array en 3 partes---------#
    part_size = len(P_tx) // 3
    P_tx_R = P_tx[:part_size]
    P_tx_G= P_tx[part_size:2*part_size]
    P_tx_B = P_tx[2*part_size:]
    
    
    
    
    print('Potencia transmitida recibida en el receptor, separada en R:', P_tx_R)
    print('Potencia transmitida recibida en el receptor, separada en G:', P_tx_G)
    print('Potencia transmitida recibida en el receptor, separada en B:', P_tx_B)
    lambda_R=7.2e-3 #(longitud de onda =630 nm)
    lambda_G=6.9e-3 #(longitud de onda =530 nm)
    lambda_B=5.6e-3 #(longitud de onda =475 nm)
    #-------------------RECEPTOR-----------------------#
    
    #--------------------Datos-------------------------#
    
    a = 3280 #resolucion pixel horizontal
    b = 2464 #resolucion pixel vertical
    c = 3.93e-5 #Aproximación Area de Tx con mitad de superficie de esfera de diametro 5mm
    f=0.00304 #Distancia focal en metros de Camaras inteligentes CMOS tipicas
    d_h=0.0036736 #Dimensión del sensor horizontal
    d_v=0.00275968 #Dimensión del sensor vertical
    a_medidapixel=d_h/a # 1.4e-6 m pixel es cuadrado 
    
    dist = 0.1  # Desde 0.1 metros hasta 14.28m, distancia entre emisor y receptor
    diam_lente = 0.004   # 4mm es el diámetro del lente óptico de una cámara que usa sensor Sony IMX219
    A_lens = np.pi * ((diam_lente / 2) ** 2)  # Área o superficie del lente
    m = 1  # coeficiente de lambertiano 1
    ang_FOV = 69.5  # Grados del campo de visión
    theta = 0  # Ángulo de emisión de la fuente 0-30
    #------------------Calculos------------------------#
    FOV_w=2*math.atan(d_h/(2*f))#campo de vision horizontal
    print(FOV_w)
    FOV_h=2*math.atan(d_v/(2*f))#campo de vision vertical
    print(FOV_w)
    N_px= c*a*b*(1/(FOV_w*FOV_h* dist ** 2)) #10% de la totalidad de pixeles
    print(N_px)
    
    
    P_rx_R = P_tx_R * (m + 1) * (1 / (2 * np.pi)) * ((np.cos(np.radians(theta))) ** m) * A_lens * np.cos(np.radians((ang_FOV))) * np.exp(-lambda_R * dist) * (1 / (dist ** 2))
    P_rx_G = P_tx_G * (m + 1) * (1 / (2 * np.pi)) * ((np.cos(np.radians(theta))) ** m) * A_lens * np.cos(np.radians((ang_FOV))) * np.exp(-lambda_G * dist) * (1 / (dist ** 2))
    P_rx_B = P_tx_B * (m + 1) * (1 / (2 * np.pi)) * ((np.cos(np.radians(theta))) ** m) * A_lens * np.cos(np.radians((ang_FOV))) * np.exp(-lambda_B * dist) * (1 / (dist ** 2))
    
    
    
    P_rx_concatenado= np.concatenate((P_rx_R, P_rx_G, P_rx_B))
    
    P_prom_pixel=P_rx_concatenado/(int(N_px))
    
    print(f"Vector de potencia P_rx: {P_prom_pixel[16588+16588:16588+165888+20]}")
    
    
    # Definimos la función para calcular E_px
    
    EQE_lambda=np.array([0.88,0.8,0.77]) #EQE para R,G,B, segun apunte OWC pin photodiodo de silicium
    # Suponiendo que tenemos un rango de longitudes de onda
    longitudes_onda = np.array([700, 530, 430])  # en nm
    E_ph_lambda=(6.626e-34 * 3e8) / (longitudes_onda * 1e-9) #Jouls
    q=1.6e-19 # Coulomb
    T_exp=85e-6 # 85 (293,663) microsegundos de exposicion
    def calcular_E_px(P_rx, EQE, E_ph, T_exp, q):
        #P_rx = P_prom_pixel=P_rx_concatenado #Vector de potencias recibidas
        # EQE: Función de eficiencia cuántica externa
        # E_ph: Energía de los fotones correspondiente a cada longitud de onda
        # T_exp: Tiempo de exposición
        # q: Carga elemental (q = 1.6e-19 C)
    
        # Calculamos la integral usando la fórmula proporcionada
        integral_resultado = np.array(P_rx * EQE * (E_ph / q))
        
        # Calculamos E_px usando la integral
        E_px = T_exp * integral_resultado
        
        return E_px
    
    # Ejemplo de cómo usar esta función
    # Supongamos que tenemos los vectores EQE y E_ph (debes definir estos vectores en base a los datos de tu sistema)
    
    
    # Calculamos E_px para el vector de potencia P_rx_concatenado
    E_px_R = calcular_E_px(P_rx_R, EQE_lambda[0], E_ph_lambda[0], T_exp, q)
    E_px_G = calcular_E_px(P_rx_G, EQE_lambda[1], E_ph_lambda[1], T_exp, q)
    E_px_B = calcular_E_px(P_rx_B, EQE_lambda[2], E_ph_lambda[2], T_exp, q)
    print(f"Energía por píxel (E_px)_R: {E_px_R}")
    print(f"Energía por píxel (E_px)_R: {E_px_G}")
    print(f"Energía por píxel (E_px)_R: {E_px_B}")
    print(len(E_px_R))
    
    E_px_conca=np.concatenate((E_px_R,E_px_G,E_px_B))
    # Definición de a_x(u) y a_y(v)
    def a_x(u, i, a, g, xi_x):
        return (u - i) * (a + g) - xi_x
    
    def a_y(v, j, a, g, xi_y):
        return (v - j) * (a + g) - xi_y
    
    # Función que se integrará
    def integrand(u, v, sigma_x, sigma_y, k):
        return np.exp(-((u**2)/(2 * sigma_x[k-1]**2) + (v**2)/(2 * sigma_y[k-1]**2)))
    
    def k_values_function(v):
        return np.arange(1, v + 1)
    
    def sigma_x_function(v):
        k_vale = np.arange(1, v + 1)  # Genera un array de NumPy desde 1 hasta v
        squared_array = np.array([x**2 for x in k_vale])  # Convertir a array de NumPy para operaciones aritméticas
        k_0 = f/dist  # CAMBIAR POR EL VALOR DE K_0 CUANDO LO ENCUENTRE
        sigma_i_prima = 1  # CAMBIAR POR EL VALOR DE sigma_i_prima CUANDO LO ENCUENTRE
        sigma_b_x = 4.22  # Valor dado en el comentario
    
        # Realizar la operación aritmética asegurando que squared_array es un array de NumPy
        squared_array_norm = sigma_b_x + squared_array * ((1 / k_0) ** 2) * sigma_i_prima
        
        return squared_array_norm
    
    def sigma_y_function(v):
        k_vale = np.arange(1, v + 1)  # Genera un array de NumPy desde 1 hasta v
        squared_array = np.array([x**2 for x in k_vale])  # Convertir a array de NumPy para operaciones aritméticas
        k_0 = f/dist  # CAMBIAR POR EL VALOR DE K_0 CUANDO LO ENCUENTRE
        sigma_i_prima = 1  # CAMBIAR POR EL VALOR DE sigma_i_prima CUANDO LO ENCUENTRE
        sigma_b_y = 4.22  # Valor dado en el comentario
    
        # Realizar la operación aritmética asegurando que squared_array es un array de NumPy
        squared_array_norm = sigma_b_y + squared_array * ((1 / k_0) ** 2) * sigma_i_prima
        
        return squared_array_norm
    
    
    def sigma_x_function_2(v):
        k_vale = np.arange(v, 0, -1)  # Genera un array de NumPy desde v hasta 1
        squared_array = np.array([x**2 for x in k_vale])  # Convertir a array de NumPy para operaciones aritméticas
        k_0 = f/dist  # CAMBIAR POR EL VALOR DE K_0 CUANDO LO ENCUENTRE
        sigma_i_prima = 1  # CAMBIAR POR EL VALOR DE sigma_i_prima CUANDO LO ENCUENTRE
        sigma_b_x = 4.22  # Valor dado en el comentario
    
        # Realizar la operación aritmética asegurando que squared_array es un array de NumPy
        squared_array_norm = sigma_b_x + squared_array * ((1 / k_0) ** 2) * sigma_i_prima
        
        return squared_array_norm
    
    
    
    
    def sigma_y_function_2(v):
        k_vale = np.arange(v, 0, -1)  # Genera un array de NumPy desde v hasta 1
        squared_array = np.array([x**2 for x in k_vale])  # Convertir a array de NumPy para operaciones aritméticas
        k_0 = f/dist  # CAMBIAR POR EL VALOR DE K_0 CUANDO LO ENCUENTRE
        sigma_i_prima = 1  # CAMBIAR POR EL VALOR DE sigma_i_prima CUANDO LO ENCUENTRE
        sigma_b_y = 4.22  # Valor dado en el comentario
    
        # Realizar la operación aritmética asegurando que squared_array es un array de NumPy
        squared_array_norm = sigma_b_y + squared_array * ((1 / k_0) ** 2) * sigma_i_prima
        
        
        return squared_array_norm
    
    def k_values_function_2(v):
        return np.arange(v, 0, -1)
    
    
    def h(x, y, i, j, a, g, xi_x, xi_y, sigma_x, sigma_y, A, c_k, k_values,k_0):
        h_sum = 0
        for k in k_values:
            h_k = (k**2 * c_k) / (2 * np.pi * sigma_x[k-1] * sigma_y[k-1] * (k_0)**2)
            print(h_k)
            integral_result = 0
            # Calcular los límites para u y v
            ax_min = a_x(x, i, a, g, xi_x)
            ax_max = a_x(x+1, i, a, g, xi_x)
            ay_min = a_y(y, j, a, g, xi_y)
            ay_max = a_y(y+1, j, a, g, xi_y)        
            integral_result, _ = quad(lambda u: quad(lambda v: integrand(x, y, sigma_x, sigma_y, k), ay_min, ay_max)[0], ax_min, ax_max)
            
            h_sum += h_k * integral_result
            #print(h_sum)
        return A * h_sum
    
    
    
    def h_centrada(x, y, i, j, a, g, xi_x, xi_y, sigma_x, sigma_y, A, c_k, k_values, k_0, W, H):
        # Calcular el centro del rectángulo
        centro_x = W / 2
        centro_y = H / 2
        
        # Desplazar las coordenadas x e y para centrar la distribución
        x_desplazado = x - centro_x
        y_desplazado = y - centro_y
        
        h_sum = 0
        for k in k_values:
            h_k = (k**2 * c_k) / (2 * np.pi * sigma_x[k-1] * sigma_y[k-1] * (k_0)**2)
            integral_result = 0
            
            # Calcular los límites para u y v utilizando las coordenadas desplazadas
            ax_min = a_x(x_desplazado, i, a, g, xi_x)
            ax_max = a_x(x_desplazado + 1, i, a, g, xi_x)
            ay_min = a_y(y_desplazado, j, a, g, xi_y)
            ay_max = a_y(y_desplazado + 1, j, a, g, xi_y)        
            
            integral_result, _ = quad(lambda u: quad(lambda v: integrand(x_desplazado, y_desplazado, sigma_x, sigma_y, k), ay_min, ay_max)[0], ax_min, ax_max)
            
            h_sum += h_k * integral_result
        
        return A * h_sum
    a_x_1=(1-b/2)*(a_medidapixel)-b/2
    a_y_1=(1-a/2)*(a_medidapixel)-a/2
    print(f" a_x  píxel(1,1) (a_x): {a_x_1}")
    print(f" a_y  píxel(1,1) (a_y): {a_y_1}")
    
    # Ejemplo de uso con las coordenadas del píxel (1,1)
    a_h = a_medidapixel  # Puedes ajustar este valor según tus necesidades
    g = 0  # Puedes ajustar este valor según tus necesidades
    xi_x = 1232
    xi_y = 1640
    i = 1232
    j = 1640
    k_0=f/dist
    
    a_x_value = a_x(1, i, a_h, g, xi_x)
    a_y_value = a_y(1, j, a_h, g, xi_y)
    print(f"a_x píxel(1,1) (a_x): {a_x_value}")
    print(f"a_y píxel(1,1) (a_y): {a_y_value}")
    
    def iteracion_h(x, y, i, j, a, g, xi_x, xi_y, A, c_k,k_0):
        # Inicializar la malla como una lista de listas (matriz 2D)
        malla = [[0 for _ in range(x)] for _ in range(y)]
    
        for x_x in range(0, x):  # Cambiado a range(0, x) para cubrir todos los índices
            for i_i in range(0, y):  # Cambiado a range(0, y) para cubrir todos los índices
                sigma_x = sigma_x_function(i_i)
                sigma_y = sigma_y_function(i_i)
                k_values = k_values_function(i_i)
                print(x_x)
                print(i_i)
                malla[x_x][i_i] = h(x_x, i_i, i, j, a, g, xi_x, xi_y, sigma_x, sigma_y, A, c_k, k_values,k_0)
    
        return malla
            
    
    # Parámetros adicionales para la función h()
    sigma_x=sigma_x_function(b)                                                   #sigma_x = np.array([1, 2, 3])  # Ejemplo de valores
    sigma_y=sigma_y_function(a)                                                   # sigma_y = np.array([1, 2, 3])  # Ejemplo de valores
    A = (1/(a_medidapixel ** 2))  # Ejemplo de valor
    c_k = 1  # Ejemplo de valor
                                                       #k_values = [1, 2, 3]  # Ejemplo de valores
    h_0 = 1  # Ejemplo de valor
    
    # Generar un array desde 1 hasta 2464
    
    u_sensor_x = np.arange(1, b+1) # 1-2465
    v_sensor_y = np.arange(1, a+1) # 1-3281
    k_values = np.arange(1,a+1)
    
    #resultado_h = h(1079, 1919, i, j, a, g, xi_x, xi_y, sigma_x_function(3280), sigma_y_function(3280), A, c_k, k_values_function(3280),k_0) # x e y van desde 0,0 (pixel 1) hasta (1079,1919) l ultimo mixel 
    #print(f"Resultado h(x,y): {resultado_h}")
    
    #h_malla=iteracion_h(2464,3280,i,j,a,g,xi_x,xi_y,A,c_k,k_0)
    #print(f"Resultado h_malla(x,y): {h_malla}")
    
    def calcular_h_en_rango(x, y,delta_x,delta_y, i, j, a, g, xi_x, xi_y, A, c_k,k_0):
        # Crear una matriz para almacenar los resultados en el rango de x y y
        resultados = []
        malla = [[0 for _ in range(x*2)] for _ in range(y*2)]
    
        # Iterar sobre el rango de x
        for x_val in np.arange(x - delta_x, x + delta_x, 1):
            fila_resultados = []
            # Iterar sobre el rango de y
            for y_val in np.arange(y - delta_y, y + delta_y, 1):
                # Calcular los valores de sigma y k_values para la coordenada y_val
                sigma_x = sigma_x_function(y_val)
                sigma_y = sigma_y_function(y_val)
                k_values = k_values_function(y_val)
    
                # Calcular h en (x_val, y_val)
                h_val = h(x_val, y_val, i, j, a, g, xi_x, xi_y, sigma_x, sigma_y, A, c_k, k_values,k_0)
                #fila_resultados.append(h_val)
                malla[x_val][y_val] = h_val
    
    
        return malla
    """
    # Ejemplo de uso
    #h_malla_entornocentro = calcular_h_en_rango(1232, 1640, 2, 2, i, j, a, g, xi_x, xi_y, A, c_k,k_0)
    #print(f"Resultado h_malla_entornocentro(x,y): {h_malla_entornocentro[1232][1640]}")
    
    
    #h_malla=iteracion_h(1232,1640,i,j,a,g,xi_x,xi_y,A,c_k,k_0)
    #print(f"Resultado h_malla(x,y): {h_malla}")
    
    resultado_h = h(0, 0, i, j, a, g, xi_x, xi_y, sigma_x_function(1), sigma_y_function(1), A, c_k, k_values_function(1),k_0) # x e y van desde 0,0 (pixel 1) hasta (1079,1919) l ultimo mixel 
    print(f"Resultado h(x,y): {resultado_h}")
    
    
    resultado_h_2 = h(0, 1, i, j, a, g, xi_x, xi_y, sigma_x_function(1), sigma_y_function(1), A, c_k, k_values_function(1),k_0) # x e y van desde 0,0 (pixel 1) hasta (1079,1919) l ultimo mixel 
    print(f"Resultado h(x,y): {resultado_h_2}")
    #mmma=max(h_malla_entornocentro)
    
    #print(mmma)
    
    #print(sigma_x_function(1000))
    
    resultado_hc_1 = h_centrada(0,0, i, j, a, g, xi_x, xi_y, sigma_x_function(1640), sigma_y_function(1640), A, c_k, k_values_function(1640),k_0,b,a) # 1232,1640 son el (0,0) en el centro de la respuesta al impulso, en ese punto k=1; entonces k va desde [-1640 (y=0), hasta 1640 (y=3280))]
    print(f"Resultado h_centrada(x,y): {resultado_hc_1}")
    
    resultado_hc_2 = h_centrada(0,3280, i, j, a, g, xi_x, xi_y, sigma_x_function(1640), sigma_y_function(1640), A, c_k, k_values_function(1640),k_0,b,a) # 1232,1640 son el (0,0) en el centro de la respuesta al impulso, en ese punto k=1; entonces k va desde [-1640 (y=0), hasta 1640 (y=3280))]
    print(f"Resultado h_centrada(x,y): {resultado_hc_2}")
    
    resultado_hc_3 = h_centrada(2464,0, i, j, a, g, xi_x, xi_y, sigma_x_function(1640), sigma_y_function(1640), A, c_k, k_values_function(1640),k_0,b,a) # 1232,1640 son el (0,0) en el centro de la respuesta al impulso, en ese punto k=1; entonces k va desde [-1640 (y=0), hasta 1640 (y=3280))]
    print(f"Resultado h_centrada(x,y): {resultado_hc_3}")
    
    resultado_hc_4 = h_centrada(2464,3280, i, j, a, g, xi_x, xi_y, sigma_x_function(1640), sigma_y_function(1640), A, c_k, k_values_function(1640),k_0,b,a) # 1232,1640 son el (0,0) en el centro de la respuesta al impulso, en ese punto k=1; entonces k va desde [-1640 (y=0), hasta 1640 (y=3280))]
    print(f"Resultado h_centrada(x,y): {resultado_hc_4}")
    """
    
    def calcular_h_centrada_en_rango(x, y, delta_x, delta_y, i, j, a, g, xi_x, xi_y, A, c_k, k_0, W, H):
        # Crear una matriz para almacenar los resultados en el rango de x y y
        resultados = []
        malla = [[0 for _ in range(2*delta_y+1)] for _ in range(2*delta_x+1)]
        matriz_final = [[0 for _ in range(2*delta_y+1)] for _ in range(2*delta_x+1)]
        # Calcular el centro del rectángulo
        centro_x = W / 2
        centro_y = H / 2
        
        # Desplazar las coordenadas x e y para centrar la distribución
        x_desplazado = x - centro_x
        y_desplazado = y - centro_y    
    
        # Iterar sobre el rango de x
        for x_val in np.arange(x_desplazado - delta_x, x_desplazado + delta_x+1, 1):
            fila_resultados = []
            # Iterar sobre el rango de y
            for y_val in np.arange(y_desplazado - delta_y, y_desplazado + delta_y+1, 1):
                if y_val==0.0:
                    sigma_x = sigma_x_function(int(abs(y_val+centro_y)))
                    sigma_y = sigma_y_function(int(abs(y_val+centro_y)))
                    k_values = k_values_function(int(abs(y_val+centro_y)))
                    print(x_val,y_val)
                
                    # Calcular h en (x_val, y_val)
                    h_val = h(x_val, y_val, i, j, a, g, xi_x, xi_y, sigma_x, sigma_y, A, c_k, k_values,k_0)
                    #fila_resultados.append(h_val)
                    malla[int(x_val)][int(y_val)] = h_val                
                    
                else:
                    # Calcular los valores de sigma y k_values para la coordenada y_val
                    sigma_x = sigma_x_function(int(abs(y_val+centro_y)))
                    sigma_y = sigma_y_function(int(abs(y_val+centro_y)))
                    k_values = k_values_function(int(abs(y_val+centro_y)))
                    print(x_val,y_val)
        
                    # Calcular h en (x_val, y_val)
                    h_val = h(x_val, y_val, i, j, a, g, xi_x, xi_y, sigma_x, sigma_y, A, c_k, k_values,k_0)
                    print(x_val, y_val)
                    #fila_resultados.append(h_val)
                    malla[int(x_val)][int(y_val)] = h_val
        for i in range(-delta_x, delta_x+1):  # i va de -5 a 5
            for j in range(-delta_y, delta_y+1):  # j va de -5 a 5
                            # Mapeo de las coordenadas: sumamos 5 para cambiar el rango de -5 a 5 a 0 a 10
                            x = i + delta_x
                            y = j + delta_y
                            matriz_final[x][y] = malla[i][j]    
        return matriz_final
    
    
    delta_x=0
    delta_y=0
        
    '''
    h_malla_centrada_entornocentro = calcular_h_centrada_en_rango(1232, 1640, delta_x, delta_y, i, j, a, g, xi_x, xi_y, A, c_k,k_0,b,a)
    print(f"Resultado h_malla_entornocentro(x,y): {h_malla_centrada_entornocentro}")
    
    # Largo en el eje x (filas)
    largo_x = len(h_malla_centrada_entornocentro)
    
    # Largo en el eje y (columnas)
    largo_y = len(h_malla_centrada_entornocentro[0])
    
    print(f"Largo en el eje x (filas): {largo_x}")
    print(f"Largo en el eje y (columnas): {largo_y}")
    
    #mmma=max(h_malla_centrada_entornocentro)
    
    #print(mmma)
    
    print(h_malla_centrada_entornocentro)
    
    
    #Ploteo de la respuesta impulsiva:
    # Convertir a array de numpy
    h_centrada_rango = np.array(h_malla_centrada_entornocentro)
    
    # Graficar usando imshow
    plt.imshow(h_centrada_rango, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Intensidad')
    plt.title('Visualización de la Matriz')
    plt.show()
    '''
    
    def reemplazar_columnas_simetricamente(malla):
        n_filas = len(malla)          # Número de filas
        n_columnas = len(malla[0])    # Número de columnas
        for i in range(n_columnas // 2):
            columna_origen = i
            columna_destino = n_columnas - i - 1
            for fila in range(n_filas):
                malla[fila][columna_destino] = malla[fila][columna_origen]
        return malla
    
    ###########-----------------H_centrada_simetrica-------------------############
    def calcular_h_centrada_en_rango_simetrico(x, y, delta_x, delta_y, i, j, a, g, xi_x, xi_y, A, c_k, k_0, W, H):
        # Crear una matriz para almacenar los resultados en el rango de x y y
        malla = [[0 for _ in range(2*delta_y+1)] for _ in range(2*delta_x+1)]
    
        # Calcular el centro del rectángulo
        centro_x = W / 2
        centro_y = H / 2
    
        # Desplazar las coordenadas x e y para centrar la distribución
        x_desplazado = x - centro_x
        y_desplazado = y - centro_y
    
        # Calcular solo para una mitad de la malla (izquierda o derecha)
        for x_val in np.arange(x_desplazado- delta_x, x_desplazado+ delta_x + 1, 1):  # Iterar solo hasta el centro
            # Iterar sobre todo el rango de y
            for y_val in np.arange(y_desplazado - delta_y, y_desplazado + delta_y + 1, 1):
                
                print(x_val)
                print(y_val)
                
                sigma_x = sigma_x_function_2(int(abs(y_val + centro_y)))
                sigma_y = sigma_y_function_2(int(abs(y_val + centro_y)))
                k_values = k_values_function_2(int(abs(y_val + centro_y)))
    
                # Calcular h en (x_val, y_val)
                h_val = h(x_val, y_val, i, j, a, g, xi_x, xi_y, sigma_x, sigma_y, A, c_k, k_values, k_0)
    
                # Colocar el valor calculado en la malla
                malla[int(x_val)][int(y_val)] = h_val
    
                # Calcular el opuesto en x (reflejar simétricamente)
              #  x_opuesto = 2 * x_desplazado - x_val  # Reflejar respecto al centro
               # if x_opuesto != x_val:  # Evitar sobreescribir el ce
                #    malla[int(x_opuesto)][int(y_val)] = h_val
                    # Devolver la matriz calculada
        
       # malla=reemplazar_columnas_simetricamente(malla)
        return malla
    
    # Definir parámetros de ejemplo (ajusta según tu modelo)
    delta_x = 0 # Ejemplo de rango en x
    delta_y = 0 # Ejemplo de rango en y
    
    
    # Ejecutar la función para calcular la malla con la simetría
    h_malla_centrada_entornocentro_simetrica = calcular_h_centrada_en_rango(1232, 1640, delta_x, delta_y, i, j, a, g, xi_x, xi_y, A, c_k, k_0, b, a)
    
    # Largo en el eje x (filas)
    largo_x = len(h_malla_centrada_entornocentro_simetrica)
    
    # Largo en el eje y (columnas)
    largo_y = len(h_malla_centrada_entornocentro_simetrica[0])
    
    print(f"Largo en el eje x (filas): {largo_x}")
    print(f"Largo en el eje y (columnas): {largo_y}")
    
    # Convertir a array de numpy
    h_centrada_rango_simetrica = np.array(h_malla_centrada_entornocentro_simetrica)
    
    
    
    """
    # Graficar usando imshow
    plt.imshow(h_centrada_rango_simetrica, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Intensidad')
    plt.title('Visualización de la Matriz Simétrica')
    #plt.show()                                                                            DESACTIVAAAAAAAAAAAAAAAAAAAR COMENTAAAAAAAAARIO
    ################---------------------##############################
    """
    
    
    
    # Supongamos que ya tienes tu matriz de 10x10
    matriz_gaussiana = h_centrada_rango_simetrica  # Reemplaza con tu variable
    
    """
    
    # Generar los índices de x e y
    x_indices = np.arange(matriz_gaussiana.shape[1])
    y_indices = np.arange(matriz_gaussiana.shape[0])
    x_mesh, y_mesh = np.meshgrid(x_indices, y_indices)
    
    # Crear la figura y los ejes 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar la superficie
    ax.plot_surface(x_mesh, y_mesh, matriz_gaussiana, cmap='viridis')
    
    # Etiquetar los ejes
    ax.set_title('Gráfico 3D de la Distribución Gaussiana')
    ax.set_xlabel('Índice X')
    ax.set_ylabel('Índice Y')
    ax.set_zlabel('Valor Gaussiano')
    
    
    # Guardar el gráfico antes de mostrarlo
    plt.savefig('grafico_gaussiano_3d.png', dpi=300)
    # Mostrar el gráfico
    #plt.show()                                                            DESACTIVAAAAAAAAAAAAAAAAAAAR COMENTAAAAAAAAARIO
    """
    
    
    ################---------------------##############################
    #Ruidos del sistema OCC
    
    i_d = 202e-6  # Corriente de oscuridad en Amperios (C/s) 1.0 * q
    # Cálculo de B
    T_delay_row=18.904e-6
    B = 1 / T_delay_row # Ancho de banda en Hz
    
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
    
    # Mostrar el resultado
    print(f"El valor de delta_th^2 es: {delta_th_squared:.4e} A^2")
    
    # Valor de q calculado anteriormente
    q = 1 / 8  # en voltios
    
    # Cálculo de sigma_adc^2
    sigma_adc_squared = (q ** 2) / 12
    
    # Mostrar el resultado
    print(f"El valor de sigma_adc^2 es: {sigma_adc_squared:.10e} V^2")
    
    delta_x=0
    delta_y=0
    
    # Límites de integración para X y Y
    u_min, u_max = 1232-delta_x, 1232+delta_x  # Definir según el dominio de interés
    v_min, v_max = 1640-delta_y, 1640+delta_y  # Definir según el dominio de interés
    x, y = 1232, 1640  # Píxel central
    
    def antesdelpixel(Gv,E_px,delta_x,delta_y):
        # Generar valores de ruido AWGN con media 0 y las varianzas especificadas
        n_th = np.random.normal(0, np.sqrt(delta_th_squared))  # Ruido térmico
        n_sh = np.random.normal(0, np.sqrt(delta_sh_squared))  # Ruido de disparo
        n_adc = np.random.normal(0, np.sqrt(sigma_adc_squared))  # Ruido ADC    
    
        # Calcular el valor de la integral (debes tener tu función 'calcular_h_centrada_en_rango' definida)
        h_matrix = calcular_h_centrada_en_rango(x, y, delta_x, delta_y, i, j, a, g, xi_x, xi_y, A, c_k, k_0, b, a)
        integral_value = sum(sum(row) for row in h_matrix)
    
        # Resultado final usando los valores de ruido generados
        resultado = Gv * (E_px * integral_value + n_th + n_sh) + n_adc
        #resultado = Gv * (E_px * integral_value)
        return resultado
    
    p_x_y_R=antesdelpixel(G_v,E_px_R,delta_x,delta_y)
    p_x_y_G=antesdelpixel(G_v,E_px_G,delta_x,delta_y)
    p_x_y_B=antesdelpixel(G_v,E_px_B,delta_x,delta_y)
    print(f"Pixel x_y_R : {p_x_y_R}")
    print(f"Pixel x_y_G : {p_x_y_G}")
    print(f"Pixel x_y_B : {p_x_y_B}")
    print(len(p_x_y_R))
    
    p_x_conca=np.concatenate((p_x_y_R,p_x_y_G,p_x_y_B))
    
    # Calcular el máximo de cada lista una vez
    max_R = max(p_x_y_R)
    max_G = max(p_x_y_G)
    max_B = max(p_x_y_B)
    
    
    list_R = [0 if val < max_R/2 else 1 for val in p_x_y_R] 
    list_G = [0 if val < max_G/2 else 1 for val in p_x_y_G]
    list_B = [0 if val < max_B/2 else 1 for val in p_x_y_B]
    
    
    # Calcular el máximo de cada lista una vez
    max_R = max(p_x_y_R)
    max_G = max(p_x_y_G)
    max_B = max(p_x_y_B)
    
    # Parámetro de n según la fórmula
    n = 6
    factor = 2 ** n  # Esto equivale a 4
    
    # Calcular el máximo de cada lista una vez
    max_R = max(p_x_y_R)
    max_G = max(p_x_y_G)
    max_B = max(p_x_y_B)
    """
    # Aplicar la fórmula a cada canal usando comprensión de listas
    list_R = [factor * (val / max_R) for val in p_x_y_R]
    list_G = [factor * (val / max_G) for val in p_x_y_G]
    list_B = [factor * (val / max_B) for val in p_x_y_B]
    print(list_R)
    
    max_list_r=max(list_R)
    max_list_g=max(list_G)
    max_list_b=max(list_B)
    
    list_R = [0 if val < 0.0284 else 1 for val in list_R]
    list_G = [0 if val < 0.0284 else 1 for val in list_G]
    list_B = [0 if val < 0.0284 else 1 for val in list_B]
    
    """
    # Concatenar las listas
    concatenated_list = list_R + list_G + list_B
    
    # Convertir la lista concatenada en un numpy array
    np_array = np.array(concatenated_list)
    
    print(np_array)
    
    T_muestreo = 1e-3  # Tiempo de muestreo en segundos
    tiempo = np.arange(0, len(list_R) * T_muestreo, T_muestreo)
    
    # Asegurarse de que el eje de tiempo tenga la misma longitud que list_R
    if len(tiempo) > len(list_R):
        tiempo = tiempo[:len(list_R)]
    """    
    # Modo interactivo activado
    plt.ion()        
    
    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(tiempo, list_R, drawstyle='steps-post', color='blue', label='list_R')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Valor de list_R')
    plt.title('Gráfica de list_R en función del tiempo')
    plt.grid(True)
    plt.legend()
    
    
    # Actualizar el gráfico sin bloquear la ejecución
    plt.draw()
    plt.pause(0.5)  # Pausa corta para actualizar la ventana
    
    #plt.show()                                                                        DESACTIVAAAAAAAAAAAAAAAAAAAR COMENTAAAAAAAAARIO
    """
    # Supongamos que concatenated_bits es el array de bits concatenados
    # bits_per_pixel = 8 (porque son 8 bits por canal para 256 niveles de color)
    bits_per_pixel = 8
    height, width = 108, 192
    total_pixels = height * width
    
    # Función para convertir 8 bits a un valor de 0 a 255
    def from_8_bits(bits):
        return np.packbits(bits)[0]
    
    # Extraer los bits correspondientes a cada canal
    # Recuerda que concatenated_bits tiene los bits de los tres canales consecutivos
    red_bits = np_array[:total_pixels * bits_per_pixel]
    green_bits = np_array[total_pixels * bits_per_pixel:2 * total_pixels * bits_per_pixel]
    blue_bits = np_array[2 * total_pixels * bits_per_pixel:]
    
    
    
    print(red_bits[0:8])
    
    # Convertir los bits de cada canal en sus valores de píxeles originales (0 a 255)
    red_channel_flat = np.array([from_8_bits(red_bits[i:i+8]) for i in range(0, len(red_bits), 8)])
    green_channel_flat = np.array([from_8_bits(green_bits[i:i+8]) for i in range(0, len(green_bits), 8)])
    blue_channel_flat = np.array([from_8_bits(blue_bits[i:i+8]) for i in range(0, len(blue_bits), 8)])
    
    print(red_channel_flat)
    
    # Reformatear los arrays planos en matrices 2D de 108x192
    red_channel = red_channel_flat.reshape((height, width))
    green_channel = green_channel_flat.reshape((height, width))
    blue_channel = blue_channel_flat.reshape((height, width))
    
    
    #Image.fromarray(red_channel, mode='L').show()  # Para mostrar solo el canal rojo             DESACTIVAAAAAAAAAAAAAAAAAAAR COMENTAAAAAAAAARIO
    #Image.fromarray(green_channel, mode='L').show()  # Para mostrar solo el canal verde
    #Image.fromarray(blue_channel, mode='L').show()  # Para mostrar solo el canal azul
    
    # Crear un array vacío con las dimensiones correctas (alto, ancho, 3)
    height, width = red_channel.shape
    reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Asignar los valores de los canales de color manualmente
    reconstructed_image[:, :, 0] = red_channel
    reconstructed_image[:, :, 1] = green_channel
    reconstructed_image[:, :, 2] = blue_channel
    
    # Convertir el array numpy a una imagen
    image_reconstructed = Image.fromarray(reconstructed_image, mode='RGB')
    
    # Guardar o mostrar la imagen reconstruida
    image_reconstructed.save('imagen_reconstruida_inversa.jpg')
    
    #image_reconstructed.show()                                                              DESACTIVAAAAAAAAAAAAAAAAAAAR COMENTAAAAAAAAARIO
    
    """
    
    def calculate_snr(G_V, E_px, delta_sh, delta_th, delta_adc):
        # Numerador de la fórmula
        numerator = (G_V**2) * (E_px**2)
        
        # Denominador de la fórmula
        denominator = (G_V**2) * (delta_sh**2 + delta_th**2) + delta_adc**2
        
        # Calcular SNR
        snr = numerator / denominator
        
        return snr
    
    def ber_ook(snr):
        # Calcular el BER en función de la SNR
        return 0.5 * erfc(np.sqrt(snr / 2))
    
    SNR=calculate_snr(3.5,np_array,delta_sh_squared,delta_th_squared,sigma_adc_squared)
    print(SNR)
    ber_snr=ber_ook(SNR)
    print(ber_snr)
    
    # Convertir SNR a dB
    SNR_dB = 10 * np.log10(SNR)
    
    
    # Submuestreo de los datos para reducir el número de puntos graficados
    factor_submuestreo = 10  # Cambia según lo que necesites
    
    SNR_dB_submuestreado = SNR_dB[::factor_submuestreo]
    ber_snr_submuestreado = ber_snr[::factor_submuestreo]
    
    # Graficar
    plt.figure(figsize=(8, 6))
    plt.plot(SNR_dB_submuestreado, ber_snr_submuestreado, label='Datos experimentales', color='blue')
    
    # Configurar la escala logarítmica para el eje y (BER)
    plt.yscale('log')
    
    # Etiquetas y título
    plt.xlabel('SNR in dB')
    plt.ylabel('BER')
    plt.title('Curvas BER vs SNR')
    
    # Añadir leyenda
    plt.legend()
    
    # Añadir grid para mejor visualización
    plt.grid(True, which="both")
    
    # Mostrar la gráfica
    plt.show()
    
    
    
    
    # Graficar
    plt.figure(figsize=(8, 6))
    plt.plot(SNR_dB, ber_snr, label='Datos experimentales', color='blue')
    
    # Configurar la escala logarítmica para el eje y (BER)
    plt.yscale('log')
    
    # Etiquetas y título
    plt.xlabel('SNR in dB')
    plt.ylabel('BER')
    plt.title('Curvas BER vs SNR')
    
    # Añadir leyenda
    plt.legend()
    
    # Añadir grid para mejor visualización
    plt.grid(True, which="both")
    
    # Mostrar la gráfica
    plt.show()
    """
    
    def cambiar_no_cero_a_uno(array):
        # Reemplazar todos los valores distintos de 0 por 1
        array[array != 0] = 1
        return array
    
    # Ejemplo de uso
    
    PrxRcambiado = cambiar_no_cero_a_uno(P_rx_R)
    List_R_array=np.array(list_R)
    
    def contar_bits_distintos(array1, array2):
        # Verificar que los arrays tengan el mismo tamaño
        if len(array1) != len(array2):
            raise ValueError("Los arrays deben tener el mismo tamaño")
    
        # Inicializar un contador
        bits_distintos = 0
        
        # Recorrer ambos arrays comparando los valores en cada índice
        for i in range(len(array1)):
            if array1[i] != array2[i]:
                bits_distintos += 1
        
        return bits_distintos
    
    # Ejemplo de uso
    
    
    # Contar la cantidad de bits distintos
    #bits_distintos = contar_bits_distintos(PrxRcambiado, List_R_array)
    
    #print(f"Cantidad de bits distintos: {bits_distintos}")
    
    def contar_diferencias(array1, array2):
        # Verificar que ambos arrays tengan el mismo tamaño
        if len(array1) != len(array2):
            raise ValueError("Los arrays deben tener el mismo tamaño para comparar")
    
        # Inicializar un contador de diferencias
        contador_diferencias = 0
    
        # Comparar elemento a elemento
        for elem1, elem2 in zip(array1, array2):
            if elem1 != elem2:
                contador_diferencias += 1
        
        return contador_diferencias
    bits_distintos = contar_diferencias(data_original, np_array)
    print(f"Cantidad de bits distintos: {bits_distintos}")
    resultados_BER.append(bits_distintos/len(data_original))
    
ejecutar_receptor('power_vector_-5.0dB.csv')
ejecutar_receptor('power_vector_-4.75dB.csv')
ejecutar_receptor('power_vector_-4.5dB.csv')
ejecutar_receptor('power_vector_-4.25dB.csv')
ejecutar_receptor('power_vector_-4.0dB.csv')
ejecutar_receptor('power_vector_-3.75dB.csv')
ejecutar_receptor('power_vector_-3.5dB.csv')
ejecutar_receptor('power_vector_-3.25dB.csv')
ejecutar_receptor('power_vector_-3.0dB.csv')
ejecutar_receptor('power_vector_-2.75dB.csv')
ejecutar_receptor('power_vector_-2.5dB.csv')
ejecutar_receptor('power_vector_-2.25dB.csv')
ejecutar_receptor('power_vector_-2.0dB.csv')
ejecutar_receptor('power_vector_-1.75dB.csv')
ejecutar_receptor('power_vector_-1.5dB.csv')
ejecutar_receptor('power_vector_-1.25dB.csv')
ejecutar_receptor('power_vector_-1.0dB.csv')
ejecutar_receptor('power_vector_-0.75dB.csv')
ejecutar_receptor('power_vector_-0.5dB.csv')
ejecutar_receptor('power_vector_-0.25dB.csv')
ejecutar_receptor('power_vector_0.0dB.csv')
ejecutar_receptor('power_vector_0.25dB.csv')
ejecutar_receptor('power_vector_0.5dB.csv')
ejecutar_receptor('power_vector_0.75dB.csv')
ejecutar_receptor('power_vector_1.0dB.csv')
ejecutar_receptor('power_vector_1.25dB.csv')
ejecutar_receptor('power_vector_1.5dB.csv')
ejecutar_receptor('power_vector_1.75dB.csv')
ejecutar_receptor('power_vector_2.0dB.csv')
ejecutar_receptor('power_vector_2.25dB.csv')
ejecutar_receptor('power_vector_2.5dB.csv')
ejecutar_receptor('power_vector_2.75dB.csv')
ejecutar_receptor('power_vector_3.0dB.csv')
ejecutar_receptor('power_vector_3.25dB.csv')
ejecutar_receptor('power_vector_3.5dB.csv')
ejecutar_receptor('power_vector_3.75dB.csv')
ejecutar_receptor('power_vector_4.0dB.csv')
ejecutar_receptor('power_vector_4.25dB.csv')
ejecutar_receptor('power_vector_4.5dB.csv')
ejecutar_receptor('power_vector_4.75dB.csv')
ejecutar_receptor('power_vector_5.0dB.csv')
ejecutar_receptor('power_vector_5.25dB.csv')
ejecutar_receptor('power_vector_5.5dB.csv')
ejecutar_receptor('power_vector_5.75dB.csv')
ejecutar_receptor('power_vector_6.0dB.csv')
ejecutar_receptor('power_vector_6.25dB.csv')
ejecutar_receptor('power_vector_6.5dB.csv')
ejecutar_receptor('power_vector_6.75dB.csv')
ejecutar_receptor('power_vector_7.0dB.csv')
ejecutar_receptor('power_vector_7.25dB.csv')
ejecutar_receptor('power_vector_7.5dB.csv')
ejecutar_receptor('power_vector_7.75dB.csv')
ejecutar_receptor('power_vector_8.0dB.csv')
ejecutar_receptor('power_vector_8.25dB.csv')
ejecutar_receptor('power_vector_8.5dB.csv')
ejecutar_receptor('power_vector_8.75dB.csv')
ejecutar_receptor('power_vector_9.0dB.csv')
ejecutar_receptor('power_vector_9.25dB.csv')
ejecutar_receptor('power_vector_9.5dB.csv')
ejecutar_receptor('power_vector_9.75dB.csv')
ejecutar_receptor('power_vector_10.0dB.csv')
ejecutar_receptor('power_vector_10.25dB.csv')
ejecutar_receptor('power_vector_10.5dB.csv')
ejecutar_receptor('power_vector_10.75dB.csv')
ejecutar_receptor('power_vector_11.0dB.csv')
ejecutar_receptor('power_vector_11.25dB.csv')
ejecutar_receptor('power_vector_11.5dB.csv')
ejecutar_receptor('power_vector_11.75dB.csv')
ejecutar_receptor('power_vector_12.0dB.csv')
ejecutar_receptor('power_vector_12.25dB.csv')
ejecutar_receptor('power_vector_12.5dB.csv')
ejecutar_receptor('power_vector_12.75dB.csv')
ejecutar_receptor('power_vector_13.0dB.csv')
ejecutar_receptor('power_vector_13.25dB.csv')
ejecutar_receptor('power_vector_13.5dB.csv')
ejecutar_receptor('power_vector_13.75dB.csv')
ejecutar_receptor('power_vector_14.0dB.csv')
ejecutar_receptor('power_vector_14.25dB.csv')
ejecutar_receptor('power_vector_14.5dB.csv')
ejecutar_receptor('power_vector_14.75dB.csv')
ejecutar_receptor('power_vector_15.0dB.csv')
ejecutar_receptor('power_vector_15.25dB.csv')
ejecutar_receptor('power_vector_15.5dB.csv')
ejecutar_receptor('power_vector_15.75dB.csv')
ejecutar_receptor('power_vector_16.0dB.csv')
ejecutar_receptor('power_vector_16.25dB.csv')
ejecutar_receptor('power_vector_16.5dB.csv')
ejecutar_receptor('power_vector_16.75dB.csv')


EbN0_dB = np.arange(-5, 17,0.25)  # Relación Eb/N0 en dB desde -10 hasta 21



# Gráfico de la curva de probabilidad de error de bits (BER)
plt.figure()
plt.semilogy(EbN0_dB, resultados_BER, 'b', label='simulation')
plt.semilogy(EbN0_dB, erfc(np.sqrt(10 ** (EbN0_dB / 10))) / 2, 'r-X', linewidth=2, label='theory')
plt.grid(True)
plt.legend()
plt.xlabel('Eb/No, dB')
plt.ylabel('Bit Error Rate')
plt.title('Bit error probability curve for OOK modulation')
# Limitar el eje y entre 10^-12 y 1
plt.ylim([1e-12, 1])
plt.show()
