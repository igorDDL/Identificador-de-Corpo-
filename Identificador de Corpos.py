
import cv2
import matplotlib.pyplot as plt
import os

# Caminho do vídeo
caminho_video = '/content/vtest.avi'

# Caminho do arquivo XML específico
caminho_classificador = '/content/haarcascade_fullbody.xml'
classificador = cv2.CascadeClassifier(caminho_classificador)

if classificador.empty():
    print(f"Erro ao carregar o classificador: {caminho_classificador}")
    exit()

# Abre o vídeo
captura = cv2.VideoCapture(caminho_video)

if not captura.isOpened():
    print(f"Erro ao abrir o vídeo: {caminho_video}")
    exit()

numero_quadro = 0

while captura.isOpened():
    sucesso, quadro = captura.read()
    if not sucesso:
        break

    quadro = cv2.resize(quadro, (640, 360))
    escala_cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)

    # Realiza a detecção
    deteccoes = classificador.detectMultiScale(
        escala_cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Desenha os retângulos ao redor das detecções
    for (x, y, l, a) in deteccoes:
        cv2.rectangle(quadro, (x, y), (x + l, y + a), (0, 255, 0), 2)
        cv2.putText(quadro, 'Pessoa', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mostrar 1 a cada 5 quadros
    if numero_quadro % 5 == 0:
        plt.imshow(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))
        plt.title(f"Quadro {numero_quadro}")
        plt.axis('off')
        plt.show()

    numero_quadro += 1

    # Processa apenas os primeiros 120 quadros
    if numero_quadro > 120:
        print("Parando após 120 quadros.")
        break

captura.release()
print("Processamento finalizado.")