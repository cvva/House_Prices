El dataset:
79 variables que describen casi cada aspecto de casas residenciales en Ames, Iowa (2006–2010). El target es SalePrice. La separación es ~1460 train / 1459 test.
Las variables más influyentes, según correlación y literatura del problema, son: OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd.

Fase 1 — EDA: lo que debes descubrir sí o sí
SalePrice tiene sesgo a la derecha — su distribución no es normal. La solución es aplicar np.log1p(SalePrice) como target. Al hacer predicciones finales aplicas np.expm1(). La métrica de la competición (RMSLE) premia exactamente esto.
Hay outliers críticos en GrLivArea: dos casas con área >4000 sqft vendidas muy baratas (granjas agrícolas). Eliminarlas en training mejora el modelo notablemente.

Fase 2 — Feature Engineering: las jugadas de alto impacto
python# Las más efectivas documentadas por la comunidad
df['TotalSF']       = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalBathrooms']= df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
df['HouseAge']      = df['YrSold'] - df['YearBuilt']
df['RemodAge']      = df['YrSold'] - df['YearRemodAdd']
df['IsNew']         = (df['YearBuilt'] == df['YrSold']).astype(int)
df['HasPool']       = (df['PoolArea'] > 0).astype(int)
df['TotalPorch']    = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

Fase 3 — Preprocesamiento: el 80% de la diferencia
Variables con muchos NaN que significan "no tiene" (no son datos faltantes reales):
pythonnone_cols = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
             'GarageType','GarageFinish','GarageQual','GarageCond',
             'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
             'MasVnrType','MSSubClass']
# → rellenar con "None"

zero_cols = ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
             'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']
# → rellenar con 0

# El resto numérico → mediana
# El resto categórico → moda
Encoding — hay variables que parecen numéricas pero son ordinales (MSSubClass, MoSold). Para las de calidad usar OrdinalEncoder con el orden correcto (Po < Fa < TA < Gd < Ex).

Fase 4 — Modelos: la combinación ganadora
El stack estándar de los notebooks top de esta competición:
ModeloPor qué funciona aquíLasso / RidgeBueno con features correlacionadas, estabilizaElasticNetCombinación L1+L2, robustoXGBoostCaptura interacciones no linealesLightGBMMás rápido que XGBoost, similar performanceGradient Boosting (sklearn)Diversidad para el ensemble
Búsqueda de hiperparámetros con Optuna o GridSearchCV con KFold(n_splits=10).

Fase 5 — Stacking / Blending: de 0.12 a 0.105
La estrategia más documentada en este problema es un weighted blend:
python# Después de tener predicciones en log-space de cada modelo
final_pred = (0.30 * xgb_pred +
              0.25 * lgbm_pred +
              0.20 * lasso_pred +
              0.15 * ridge_pred +
              0.10 * elasticnet_pred)

submission['SalePrice'] = np.expm1(final_pred)
Los pesos se optimizan con scipy.optimize.minimize minimizando RMSLE en out-of-fold predictions.

Variables que la comunidad identifica como más dañinas
Utilities (casi todas las casas tienen el mismo valor), StreetCondition, PoolQC (muy poca varianza). Eliminarlas o dejar que Lasso las ponga a cero solo.

Estrategia de ataque recomendada para ti
Dado tu stack de Python/pandas/sklearn, te recomiendo este orden:

Notebook de EDA con histogramas, heatmap de correlación, boxplots por Neighborhood
Pipeline sklearn con ColumnTransformer para separar numéricas y categóricas limpiamente
Modelo base con Ridge + log-target para tener un baseline sólido (~0.13)
Agregar LightGBM + XGBoost con Optuna
Blend final de 4-5 modelos