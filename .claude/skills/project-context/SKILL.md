---
name: project-context
description: Carga el contexto del proyecto Kaggle "House Prices - Advanced Regression Techniques". Úsala al inicio de una sesión o cuando necesites situarte en el estado actual antes de continuar. Lee SOLO las fuentes canónicas del proyecto -notebook principal pca_houses_prices_kaggle.ipynb, datasets, README y dependencias- y produce un resumen estructurado, sin escanear notebooks antiguas ni carpetas de exploración.
---

# Contexto del proyecto: House Prices (Kaggle)

Objetivo de esta skill: reconstruir rápidamente el estado actual del proyecto leyendo únicamente las fuentes autorizadas y estructurar el roadmap de mejoras.

## Fuentes canónicas (leer en este orden)

1. **`README.md`** — Propósito, alcance y estructura declarada del proyecto.
2. **Dependencias** — `pyproject.toml` y `requirements.txt` (versiones de pandas, scikit-learn, xgboost, lightgbm, optuna).
3. **`data/data_description.txt`** — Diccionario de variables del dataset (para consulta semántica bajo demanda; no resumir completo).
4. **Datasets** en `data/`: `train.csv` (1,460 filas), `test.csv` (1,459 filas) y `sample_submission.csv`. Obtén forma y columnas sin volcar los archivos completos.
5. **Notebook principal `pca_houses_prices_kaggle.ipynb`** — Notebook unificada que contiene el pipeline de alto rendimiento actual. Extrae el texto de las celdas (markdown + código) omitiendo los outputs para optimizar contexto.
6. **Notebooks de referencia en `claude/`** — Desglose paso a paso del pipeline del flujo `house_prices_pipeline.png`:
   - `1_house_prices_EDA.ipynb`: Análisis exploratorio profundo, target log-transform, identificación de outliers.
   - `2_house_prices_feature_engineering.ipynb`: Creación de features de área, calidad, antigüedad, y Target Encoding de vecindario.
   - `3_house_prices_preprocessing.ipynb`: Integridad, corrección de skewness, escalado RobustScaler y preselección con Lasso.
   - `4_house_prices_modeling.ipynb`: Modelos base (Ridge, Lasso, ElasticNet) y optimización de XGBoost/LightGBM con Optuna.
   - `5_house_prices_stacking_blending.ipynb`: Análisis de correlación de OOF predictions, optimización de pesos y meta-modelado.

## Resumen Técnico del Estado Actual (`pca_houses_prices_kaggle.ipynb`)

El pipeline unificado implementa las siguientes fases estratégicas:
- **Tratamiento del Target**: `log1p(SalePrice)` para corregir asimetría.
- **Feature Engineering (80+ variables nuevas)**:
  - *Áreas y Porches*: `TotalSF` (Sótano + 1st + 2nd Flr), `TotalAbvGrdSF`, `TotalPorchSF` (porches + deck), `BsmtToTotalRatio`, `SecFloorRatio`, `LotDensity`.
  - *Baños*: `TotalBathrooms` (dando peso de 0.5 a los medios baños), `AbvGrdBathrooms`, `BathRatio`.
  - *Calidades Compuestas*: `QualityScore` (ponderado de calidades exteriores/interiores), `ConditionScore`, `QualCondIndex`, `TotalQualSF` y `GarageScore`.
  - *Temporales*: `HouseAge`, `RemodAge`, `GarageAge`, `IsNew`, `WasRemodeled`, `EffectiveAge` y `EraBuilt` (binned).
  - *Flags Binarios*: Presencia de piscina, garaje, sótano, chimenea, deck, etc.
  - *Interacciones*: `OverallQual * TotalSF`, `OverallQual * GrLivArea`, `OverallQual * YearBuilt`, etc.
  - *Numéricas Transformadas*: Cuadrados de variables clave y logaritmo `log1p` para variables de área altamente sesgadas.
  - *Encoding Ordinal*: Mapeo explícito a escalas numéricas de 9 variables categóricas ordinales.
- **Preprocesamiento**:
  - Imputación inteligente de `LotFrontage` usando la mediana de su vecindario (`Neighborhood`).
  - Target Encoding con K-Fold (5-fold, `smoothing=10`) en variables categóricas de alta cardinalidad para evitar fuga de información: `Neighborhood`, `MSSubClass`, `MSZoning`, `SaleType`, `SaleCondition`.
  - Corrección de asimetría (`skewness > 0.75`) aplicando `log1p` a 53 variables continuas.
  - Escalado robusto (`RobustScaler`) para modelos lineales; dataset original codificado para modelos basados en árboles.
- **Modelos Base y Validación**:
  - Esquema de CV: 10-Fold estratificado por aleatoriedad.
  - XGBoost y LightGBM optimizados usando Optuna (50 trials cada uno en K-Fold de 5 divisiones).
- **Resultados de Validación Cruzada (OOF RMSLE)**:
  - **Ridge** (`alpha=15`): `0.12694`
  - **ElasticNet** (`alpha=0.001`, `l1=0.9`): `0.12688`
  - **Lasso** (`alpha=0.0003`): `0.12388`
  - **LightGBM** (Optuna): `0.12185`
  - **XGBoost** (Optuna): `0.11808` (Mejor modelo base)
- **Ensamble Final (Blending)**:
  - Optimización de pesos vía SLSQP: **XGBoost (0.6380) + Lasso (0.3620)**.
  - **Ensemble OOF RMSLE**: **`0.11522`** (Mejora absoluta de `+0.00286` frente al mejor modelo base).
  - Brecha actual al **Top 5% (`< 0.105`)**: **`~10.2` puntos de RMSLE**.

## Comandos de extracción eficientes

Para ejecutar localmente, utiliza el Python del entorno virtual: `.venv/Scripts/python`.

1. **Obtener formas y nulos de los datasets**:
```bash
.venv/Scripts/python -c "
import pandas as pd, sys; sys.stdout.reconfigure(encoding='utf-8')
for n in ['train','test','sample_submission']:
    df = pd.read_csv(f'data/{n}.csv')
    print(f'{n}: {df.shape}')
"
```

2. **Extraer el código y markdown sin outputs de la notebook principal**:
```bash
.venv/Scripts/python -c "
import json, sys; sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open('pca_houses_prices_kaggle.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    print(f'--- [{i}] {c[\"cell_type\"]} ---'); print(s)
"
```

## Plan de Mejoras Estratégicas para alcanzar el Top 5% (< 0.105)

Si el objetivo es optimizar y refinar el modelo, sugiere e implementa los siguientes pasos en la sesión de trabajo:
1. **Diversidad de Modelos (Stacking)**: Incorporar nuevos estimadores no lineales de alta calidad al blend:
   - **CatBoost Regressor** (extremadamente potente con categóricas e insensible al sobreajuste).
   - **Random Forest** y **Extra Trees** de scikit-learn para agregar diversidad de estimación (fases EDA de bagging).
   - **Gradient Boosting Regressor** de scikit-learn.
2. **Optuna para Modelos Lineales**: Realizar tuning de hiperparámetros finos (`alpha` en Lasso y Ridge; `alpha` y `l1_ratio` en ElasticNet) mediante Optuna en lugar de usar constantes fijas.
3. **Limpieza Avanzada de Outliers**:
   - Identificar outliers multivariantes en el espacio de entrenamiento mediante análisis de residuos del modelo base (los registros con mayor error de predicción en OOF).
   - Probar umbrales de remoción específicos (p. ej., eliminar propiedades con residuos extremos).
4. **Optimización de Target Encoding**:
   - Tunear el parámetro de `smoothing` (actualmente en 10) usando Optuna.
   - Experimentar con Target Encoding en más variables categóricas medianas.
5. **Reducción de Dimensionalidad / Multicolinealidad**:
   - Analizar correlación entre features de feature engineering.
   - Remover features con correlación extrema (`> 0.95`) que aportan ruido o redundancia a los modelos lineales (Lasso e ElasticNet reducen coeficientes, pero un filtrado previo ayuda a la estabilidad).

## Qué entregar (resumen estructurado al cargar la skill)

Tras cargar la skill, produce un reporte con estas secciones:
1. **Estado del Pipeline**: Confirmar que `pca_houses_prices_kaggle.ipynb` es el archivo canónico.
2. **Resultados de Métricas**: Listar los últimos valores de OOF RMSLE y pesos del ensemblado.
3. **Diagnóstico**: Explicar la brecha restante hacia el target de `< 0.105`.
4. **Propuesta del Siguiente Paso**: Proponer una de las mejoras estratégicas (p. ej., "Paso 1: Integrar CatBoost con Optuna y recalcular el blending").
