import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

def kmeans(imgIn, k):
  imgRGB = cv2.cvtColor(imgIn, cv2.COLOR_GRAY2RGB)
  imgPix = imgRGB.reshape((-1,3)) 

  imgPix = np.float32(imgPix)

  _, pixels, colors = cv2.kmeans(
    imgPix,
    k,
    None,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85),
    2,
    cv2.KMEANS_RANDOM_CENTERS
  )

  colors = np.uint8(colors)[:,0]
  dataSeg = colors[pixels.flatten()]
  imgSeg = dataSeg.reshape(imgIn.shape)
  colors.sort()

  return colors, imgSeg


GLAND_DENS = 0
GLAND_FATTY = 1
FATTY = 2

def mamSegClas(fname):
  imgIn = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

  # Se segmenta la imagen con kmeans
  colors, imgSeg = kmeans(imgIn, 5)

  # Se umbraliza la imagen segmentada con el color del grupo mas claro en el que se encuentra el musculo
  _, imgMusc = cv2.threshold(imgSeg, colors[4]-1, 255, cv2.THRESH_BINARY)

  # Se aplica un suavizado morfologico para eliminar detalles irrelevantes y separar los contornos
  imgMusc = cv2.morphologyEx(
    imgMusc,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)),
    iterations=1
  )

  imgMusc = cv2.morphologyEx(
    imgMusc,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)),
    iterations=5
  )

  # Se obtienen los contornos del grupo mas claro
  cntsMusc, _ = cv2.findContours(imgMusc, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  cntMusc = None
  maxArea = 0
  right = True

  # Se identifica el contor del musculo en base al angulo y area
  for cnt in cntsMusc:
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    area = cv2.contourArea(cnt)

    if topmost[1] < 100 and area > 100:
      _,_,ang =cv2.fitEllipse(cnt)

      if 10 < ang < 60 or 150 < ang < 200:
        if maxArea < area:
          cntMusc = cnt
          right = 150 < ang < 200

  # Se dibuja la mascara del musculo que se usara mas adelante
  maskMusc = np.zeros(imgIn.shape).astype(imgIn.dtype)

  if cntMusc is not None:
    cv2.drawContours(maskMusc, [cntMusc], -1, 255, cv2.FILLED)
    maskMusc = cv2.morphologyEx(
      maskMusc,
      cv2.MORPH_DILATE,
      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
      iterations=10
    )

  # Se umbraliza la imagen segmentada con el color del grupo mas oscuro para quitar el fondo
  _, imgMama = cv2.threshold(imgSeg, colors[0], 255, cv2.THRESH_BINARY)

  # Se aplica un suavizado morfologico para eliminar detalles irrelevantes y separar los contornos
  imgMama = cv2.morphologyEx(
    imgMama,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)),
    iterations=1
  )

  imgMama = cv2.morphologyEx(
    imgMama,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)),
    iterations=1
  )

  # Se obtienen los contornos del grupo mas oscuro
  cntsMama, _ = cv2.findContours(imgMama, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  # Se identifica la mama en base al area
  maxArea = 0
  for cnt in cntsMama:
    area = cv2.contourArea(cnt)
    if maxArea < area:
      maxArea = area
      cntMama = cnt

  # Se dibuja la mascara de la mama que se usara a continuacion
  maskMama = np.zeros(imgIn.shape).astype(imgIn.dtype)
  cv2.drawContours(maskMama, [cntMama], -1, 255, cv2.FILLED)

  # Se consigue la mascara de solo la mama restandole el musculo
  maskMamaNotMusc = cv2.bitwise_and(maskMama, cv2.bitwise_not(maskMusc))

  # Se le aplica un suavizado morfolofico para eliminar detalles irrelevantes
  maskMamaNotMusc = cv2.morphologyEx(
    maskMamaNotMusc,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)),
    iterations=2
  )

  maskMamaNotMusc = cv2.morphologyEx(
    maskMamaNotMusc,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
    iterations=1
  )

  # Se dibuja el perimetro de la mama
  cntsMamaNotMusc, _ = cv2.findContours(
    maskMamaNotMusc,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_NONE
  )

  assert len(cntsMamaNotMusc) == 1

  imgOut = cv2.drawContours(imgIn.copy(), cntsMamaNotMusc, -1, 255, cv2.LINE_4)


  # Se calcula el tamano de la mama en base al area total de la imagen
  mamSize = np.count_nonzero(maskMamaNotMusc)

  mamProp = mamSize/(imgIn.shape[0]*imgIn.shape[1])


  # Se calcula la proporcion de glandulas en la mama
  imgGland = np.zeros(imgIn.shape).astype(imgIn.dtype)
  cv2.drawContours(imgGland, cntsMusc, -1, 255, cv2.FILLED)
  imgGland = cv2.bitwise_and(imgGland, maskMamaNotMusc)

  glandProp = np.count_nonzero(imgGland)/mamSize


  # Se calcula la proporcion de fibras en la mama
  imgCutMama = cv2.bitwise_and(imgIn, maskMamaNotMusc)

  imgCutMama = cv2.equalizeHist(imgCutMama)

  imgFib = cv2.Canny(imgCutMama, 250, 100)

  imgFib = cv2.morphologyEx(
    imgFib,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,3)),
    iterations=1
  )

  imgFib = cv2.morphologyEx(
    imgFib,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,3)),
    iterations=1
  )

  fibProp = np.count_nonzero(imgFib)/mamSize


  # Se clasifica en glandular-densa, glandular-grasa, y grasa
  mamPropW = 0.2
  fibPropW = 0.3
  glandPropW = 0.5

  glandDensePts = 0
  glandFattyPts = 0
  fattyPts = 0

  glandDensePts += abs(mamProp-0.3211946487426758)*mamPropW
  glandFattyPts += abs(mamProp-0.3181273937225342)*mamPropW
  fattyPts += abs(mamProp-0.33791375160217285)*mamPropW

  glandDensePts += abs(fibProp-0.01459385936678027)*fibPropW
  glandFattyPts += abs(fibProp-0.027576374055264568)*fibPropW
  fattyPts += abs(fibProp-0.06562819896057095)*fibPropW

  glandDensePts += abs(glandProp-0.28682820272659826)*glandPropW
  glandFattyPts += abs(glandProp-0.10679748607965273)*glandPropW
  fattyPts += abs(glandProp-0.02427641694168573)*glandPropW

  if glandDensePts < glandFattyPts and glandDensePts < fattyPts:
    classRes = GLAND_DENS
  elif glandFattyPts < glandDensePts and glandFattyPts < fattyPts:
    classRes = GLAND_FATTY
  elif fattyPts < glandDensePts and fattyPts < glandFattyPts:
    classRes = FATTY
  else:
    classRes = -1

  return imgOut, mamProp, fibProp, glandProp, classRes


def main(argc, argv):
  assert argc > 1

  mamPropList = np.empty(argc-1)
  fibPropList = np.empty(argc-1)
  glandPropList = np.empty(argc-1)

  i = 0
  for fname in argv[1:]:
    imgOut, mamProp, fibProp, glandProp, classRes = mamSegClas(fname)

    plt.figure(fname)
    plt.imshow(imgOut, cmap="gray")
    plt.show(block=False)

    print("Clasificación de la imagen {}:".format(fname))
    print()
    print(
      "Glandular densa" if classRes == GLAND_DENS else
      "Glandular grasa" if classRes == GLAND_FATTY else
      "Grasa" if classRes == FATTY else
      "Ninguna"
    )
    print()
    print()

    mamPropList[i] = mamProp
    fibPropList[i] = fibProp
    glandPropList[i] = glandProp

    i += 1

  print("Resultados globales:")
  print()
  print("MAMA")
  print("mean")
  print(mamPropList.mean())
  print("std")
  print(mamPropList.std())
  print()
  print("FIBER")
  print("mean")
  print(fibPropList.mean())
  print("std")
  print(fibPropList.std())
  print()
  print("GLANDS")
  print("mean")
  print(glandPropList.mean())
  print("std")
  print(glandPropList.std())

  plt.show()
  

if __name__ == '__main__':
  main(len(sys.argv), sys.argv)