# TP RN Regularización Bonacina

Este TP se construye intencionalmente sobre la base del TP de Softmax, según lo indicado por Bonacina. Se preservó el notebook inicial sin modificar código ni resultados. El desarrollo de Regularización continuará sobre esta base; al incorporar sus experimentos se utilizarán nombres propios para distinguir sus registros de los del TP de Softmax. Se incluye arquitectura-tp-regularizacion.html como material de apoyo. No se copiaron registros de otros TPs como si fueran propios.

## Archivos y ejecución

- Notebook principal: `TP_Regularización.ipynb`.
- Dependencias con versiones: `requirements.txt`. Versiones observadas: `entorno-origen.json`.
- Ejecutar `python -m pip install -r requirements.txt` y abrir el notebook con ese entorno en Jupyter o VS Code, usando esta carpeta como directorio de trabajo.
- La primera ejecución descarga 20 Newsgroups y recursos de NLTK y requiere conexión a Internet.
- Separar los archivos no volvió a ejecutar entrenamientos ni verificó una instalación limpia. Los resultados guardados se pueden leer sin entrenar.

## Registros de los experimentos

Registros de MLflow conservados: **0 corridas**. `mlflow.db` contiene únicamente el experimento de este TP. `registros/mlflow/` ofrece las tablas de corridas, parámetros, métricas por paso y últimas métricas en CSV. Los gráficos disponibles se conservan en `mlruns/` y los eventos de TensorBoard en `runs/`.

Para revisar MLflow ejecutar `python ver_registros.py` y abrir la dirección local que indique. Este comando actualiza las rutas de artefactos al directorio actual. Para TensorBoard ejecutar `tensorboard --logdir runs`.

Los modelos entrenados y su preprocesamiento se conservan localmente en `modelos/` cuando estaban disponibles y se excluyen de GitHub. No se entregan los pesos de modelos de MLflow. Las carpetas `runs/`, `mlruns/`, los CSV y la base `mlflow.db` sí se versionan, conforme a la indicación del profesor del 4 de noviembre.

## Procedencia

Separado el 20 de noviembre de 2026 desde https://github.com/DinosaurioBlue/RN_Bonacina_TPs, revisión `6f8566054ae62b94b7cb2c9a09620bc806a473d9`. Los notebooks y los archivos copiados se verificaron por SHA-256; se conservaron sin cambios. El repositorio original y los experimentos auxiliares permanecen en `01 - TP's` como respaldo. Los historiales anteriores continúan disponibles en el repositorio original.
