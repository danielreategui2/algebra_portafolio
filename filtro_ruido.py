# ============================================
# PROYECTO: MEJORA DE IMAGENES CON MATRICES
# ALGEBRA LINEAL
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. CARGAR IMAGEN COMO MATRIZ
nombre = "pareja.jpg"   # misma carpeta que este archivo
imagen = Image.open(nombre).convert("L")
A = np.array(imagen, dtype=np.float64)
print("Dimensiones de la matriz A:", A.shape)

# 2. MEJORA DE CONTRASTE: transformacion lineal  B = a*A + b
#    Se estiran los valores entre los percentiles 1 y 99 al rango 0-255
minimo, maximo = np.percentile(A, 1), np.percentile(A, 99)
a = 255 / (maximo - minimo)
b = -a * minimo
A_contraste = np.clip(a * A + b, 0, 255)
print(f"Transformacion lineal: B = {a:.4f} * A + ({b:.2f})")

# 3. KERNEL DE NITIDEZ 3x3 (realza bordes y detalles)
#    K = I + Laplaciano  ->  suma de entradas = 1 (conserva el brillo)
K = np.array([[ 0, -1,  0],
              [-1,  5, -1],
              [ 0, -1,  0]], dtype=np.float64)
print("\nMatriz del filtro de nitidez K:")
print(K)

# 4. APLICAR EL KERNEL (producto elemento a elemento y suma)
def aplicar_kernel(matriz, kernel):
    filas, columnas = matriz.shape
    resultado = np.zeros_like(matriz)
    imagen_padded = np.pad(matriz, 1, mode="edge")   # conserva dimensiones
    for i in range(filas):
        for j in range(columnas):
            region = imagen_padded[i:i+3, j:j+3]
            resultado[i, j] = np.sum(region * kernel)
    return resultado

A_mejorada = np.clip(aplicar_kernel(A_contraste, K), 0, 255)

# 5. METRICAS (no hay imagen "perfecta" de referencia, por eso se mide
#    nitidez y contraste en lugar de PSNR)
K_lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

def nitidez(m):                      # varianza del Laplaciano
    return aplicar_kernel(m, K_lap).var()

def contraste(m):                    # desviacion estandar de intensidades
    return m.std()

# 6. MOSTRAR RESULTADOS
fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
titulos = ["Imagen original", "Con contraste", "Contraste + nitidez"]
for eje, m, t in zip(ejes, [A, A_contraste, A_mejorada], titulos):
    eje.imshow(m, cmap="gray", vmin=0, vmax=255)
    eje.set_title(t)
    eje.axis("off")
plt.tight_layout()
plt.savefig("comparacion.png", dpi=150)

# 7. MOSTRAR METRICAS
print("\nRESULTADOS DEL PROCESAMIENTO")
print("--------------------------------")
print(f"{'':22}{'Nitidez':>12}{'Contraste':>12}")
for nom, m in [("Original", A), ("Con contraste", A_contraste), ("Contraste + nitidez", A_mejorada)]:
    print(f"{nom:22}{nitidez(m):12.2f}{contraste(m):12.2f}")

# 8. EJEMPLO MANUAL 5x5 (para comparar con el calculo a mano)
B = np.array([[52, 55, 61, 66,  70],
              [63, 59, 55, 90, 109],
              [85, 104, 92, 79, 70],
              [71, 60, 64, 100, 82],
              [69, 70, 68, 60,  75]], dtype=np.float64)
R = aplicar_kernel(B, K)
print("\nEJEMPLO 5x5 - resultado del filtro (sin recortar):")
print(R)
print("Pixel central (2,2):", R[2, 2], "= 5*92 - (55+104+79+64)")

# 9. GUARDAR IMAGEN MEJORADA
Image.fromarray(A_mejorada.astype(np.uint8)).save("imagen_mejorada.png")
print("\nImagen guardada como imagen_mejorada.png")

plt.show()