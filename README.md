# TP3 · Regularización en 20 Newsgroups

Comparamos L1, L2, L1+L2, Dropout y BatchNorm sobre una MLP con LeakyReLU
(pendiente 0,1). Usamos Adam, Xavier uniforme y semilla 3 por defecto.

Tarde demasiado en realizar estos tps, ya que no estuve muy conformo con mis resultados y metodologías anterioriores y rehice todo el modelo desde 0 reviendo las clases.

## Notebook

- **0 a 5.** Setup, datos, preprocesamiento, TF-IDF, modelo y entrenamiento.
  Reservamos cerca del 20 % del train oficial para dev, agrupando textos
  repetidos. El TF-IDF se ajusta con train (unigramas, hasta 5000 términos,
  `min_df=1`, norma L2).
- **6. Baseline.** Una capa de 256 neuronas, batch 128 y LR fijo de 0,00245.
  Guardamos la mejor época en dev y la última. Fue tomado a partir de uno de los
  mejores modelos del tp2, intentando que "overfitie" más aún y con más ruido.
- **7. Regularización.** Mantenemos la arquitectura, el LR y los pesos
  iniciales del baseline. Miramos curvas, variación de pérdida entre batches,
  magnitud de los pesos y tiempos. En 7.6 probamos early stopping en el baseline
  y en 7.7 lo agregamos a la combinación. La paciencia es de 15 épocas.

- **8. Optuna.** 24 trials con LR fijo y 24 con CLR, hasta 40 épocas por
  trial. Probamos 1, 2 o 3 capas, con 128, 256 o 512 neuronas por capa y
  batch 64, 128 o 256. Cada trial usa una opción en todas las capas ocultas:
  ninguna, BatchNorm o Dropout. Cada opción tiene un trial inicial por
  profundidad. Si se usa Dropout, su probabilidad va de 0,1 a 0,6 en pasos de 0,1.
  Corremos sin poda para comparar las curvas completas. Elegimos por accuracy de dev.
- **9. Test.** Evaluamos el baseline, el mejor con LR fijo, el mejor con CLR
  y el mejor entre BatchNorm y Dropout. Si coinciden dos puestos, usamos un solo modelo.
  Mostramos el test oficial y la accuracy quitando las copias del train
  oficial.

## CLR con Adam

Smith propone subir y bajar el LR por batch. Con `triangular2`, el tamaño
de la oscilación se reduce a la mitad al terminar cada ciclo. En un ensayo de
CIFAR-10, obtuvo 81,4 % de accuracy en 25.000 iteraciones, frente a 70.000 con
LR fijo. [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)

### Cooking recipe

1. Arrancar con un baseline nuevo. Subir el LR linealmente de `1e-5` a
   `1e-2` durante 6 épocas y medir accuracy y pérdida en dev cada media época.
2. Tomar como techo la primera caída de al menos 1,5 puntos después del pico
   de accuracy, o el último LR medido. Como referencia para el mínimo, usar
   el primer punto a menos de 2 puntos del pico, limitado a un tercio del techo.
3. Reiniciar pesos y Adam. Usar dos épocas de subida y dos de bajada con
   `triangular2`. El medio ciclo es `2 * len(train_loader)`. Ejecutar
   `scheduler.step()` después de cada `optimizer.step()`, con
   `cycle_momentum=False`.
4. Con esos límites de referencia, Optuna elige la base y el pico inicial
   de CLR. El estudio con LR fijo también explora valores bajos. Comparar
   las configuraciones en dev antes de abrir el test.

## Ejecución desde un clon

Abrir `TP_Regularización.ipynb` desde la carpeta del repo y ejecutar las
secciones 0 a 9 en orden. La primera vez hace falta internet para descargar
20 Newsgroups, los recursos de NLTK y los paquetes del setup. Para las
secciones 8 y 9, PyTorch tiene que detectar una GPU (`cuda` o `xpu`).
En VS Code usé el kernel `mlp-gpu` con `xpu`. En Colab hay que activar
GPU y montar Drive cuando lo pide el notebook.

La corrida original de semilla 3 está incluida en
[.experimentos/my_gpu_search/ba3ca4d5af94](.experimentos/my_gpu_search/ba3ca4d5af94).
Contiene los estudios de Optuna, las métricas, las predicciones y los
modelos finales. Las dos celdas de resumen de las secciones 8 y 9 leen
esa carpeta. Si se hace una corrida nueva, se pueden saltear esas dos
celdas. Las tablas anteriores de cada sección muestran los resultados
de la nueva corrida. Si Optuna encuentra el estudio guardado en la misma
ruta, lo reutiliza. Para empezar de cero, mover temporalmente esa
carpeta fuera del clon antes de ejecutar.

La semilla 42 fija la partición y la 3 se usa para la inicialización,
la mezcla de batches, Dropout y Optuna. Aun con la misma semilla, algunos
cálculos en GPU pueden variar según el hardware, los drivers y la versión
de PyTorch. Diferencias pequeñas pueden cambiar qué época queda primera
en dev y qué configuración termina ganando. Las tablas y los gráficos
guardados en el notebook corresponden a la corrida original.
