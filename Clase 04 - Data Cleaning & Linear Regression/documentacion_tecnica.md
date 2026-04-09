# README Técnico — Clase04_DataCleaning_RegresionLineal.ipynb

Documentación del código celda por celda: qué hace cada bloque, qué produce y por qué.

> **Para qué sirve este archivo:** el notebook está pensado para ser ejecutado en clase y queda intencionalmente "limpio" (solo código y observaciones cortas). Si querés entender en detalle qué hace cada línea de código, leé este archivo. Si querés entender qué encontramos en los datos y qué decisiones tomamos, leé `hallazgos.md`.

## Estructura del notebook (CRISP-DM)

El notebook sigue las fases del proceso estándar de un proyecto de datos:

| Fase CRISP-DM | Sección del notebook |
|---|---|
| **Business Understanding** | Markdown inicial — pregunta de negocio |
| **Data Understanding** | A.1 (diagnóstico) |
| **Data Preparation** | A.2 → A.7 (limpieza) |
| **Modeling** | B.1 → B.4 |
| **Evaluation** | B.5 → B.7 |
| *Deployment* | Fuera del scope de la clase |

---

## Stack de dependencias

```python
pandas       # manipulación de datos (DataFrames, fillna, str.replace, drop_duplicates, etc.)
numpy        # operaciones numéricas (sqrt, linspace, quantile usado por pandas)
matplotlib   # renderizado base de gráficos (pyplot, axes, fig)
seaborn      # gráficos estadísticos de alto nivel (boxplot, heatmap)
scikit-learn # modelado: train_test_split, LinearRegression, métricas MSE / R²
```

Versiones recomendadas: `pandas >= 2.0`, `scikit-learn >= 1.3`, `seaborn >= 0.12`.

---

## 0. Setup

### Celda 1 — Imports, configuración global y carga del dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style("whitegrid")

df = pd.read_csv("campañas_marketing.csv")
print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")
df.info()
df.head()
```

| Bloque | Efecto |
|---|---|
| `import pandas as pd` (y los demás) | Carga las librerías con sus alias estándar (`pd`, `np`, `plt`, `sns`). |
| `from sklearn...` | Importa solo lo que vamos a usar de scikit-learn: el splitter, el modelo y las dos métricas. Es buena práctica importar lo justo y necesario en lugar de `from sklearn import *`. |
| `sns.set_style("whitegrid")` | Aplica un estilo visual global (fondo blanco con grilla gris) a todos los gráficos posteriores. Solo hay que ejecutarlo una vez. |
| `pd.read_csv("campañas_marketing.csv")` | Lee el CSV y lo carga como un DataFrame en la variable `df`. Asume que el archivo está en el mismo directorio que el notebook. |
| `df.shape[0]` / `df.shape[1]` | Cantidad de filas y columnas, respectivamente. `shape` devuelve una tupla `(filas, columnas)`. |
| `df.info()` | Imprime el tipo de dato de cada columna y cuántos valores no son nulos. **Es el primer chequeo de salud del dataset.** Si una columna numérica aparece como `object`, hay un problema. |
| `df.head()` | Muestra las primeras 5 filas. Como es la última expresión de la celda, se renderiza automáticamente como tabla. |

> Acá ya se ven dos cosas raras: `inversion_usd` aparece como `object` (es texto en lugar de número), y hay menos de 93 valores no-nulos en `inversion_usd` y `clicks` → faltantes.

---

## Parte A — Data Cleaning

### Celda 2 — Diagnóstico A.1: nulos, duplicados y categorías inconsistentes

```python
# Faltantes, categorías inconsistentes y duplicados
print("Nulos por columna:")
print(df.isnull().sum())
print(f"\nFilas duplicadas: {df.duplicated().sum()}")
print("\nCanales únicos:")
print(df["canal"].value_counts())
print("\nRegiones únicas:")
print(df["region"].value_counts())
```

| Bloque | Efecto |
|---|---|
| `df.isnull()` | Genera un DataFrame de True/False del mismo tamaño: True donde hay un nulo. |
| `.sum()` | Suma los True por columna (True = 1, False = 0). El resultado es la cantidad de nulos por columna. |
| `df.duplicated().sum()` | Cuenta filas que son duplicado exacto de una fila anterior (la primera ocurrencia no se cuenta). El fix ocurre en A.6, pero lo detectamos acá para que el alumno vea **los 5 problemas en una sola pasada** (Data Understanding). |
| `df["canal"].value_counts()` | Cuenta cuántas veces aparece cada valor en la columna `canal`, ordenado de mayor a menor. **Si una categoría se repite con diferentes mayúsculas/espacios/sinónimos, los va a listar como categorías distintas → eso revela las inconsistencias.** |
| `\n` en los `print` | Es un salto de línea — separa visualmente los bloques en el output. |

### Celda 3 — Diagnóstico A.1: rangos numéricos (`describe`)

```python
# Rangos de las variables numéricas
df.describe()
```

`.describe()` devuelve para cada columna numérica: count, mean, std, min, 25%, 50% (mediana), 75%, max. Es la primera lectura cuantitativa del dataset y sirve para detectar valores extremos: si el `max` está muy por encima del Q3, hay un candidato a outlier.

> **Observación esperada:** el `max` de `ventas_usd` (~$45.000) está muy por encima del Q3 (~$6.400). Lo confirmamos visualmente con un boxplot en la celda siguiente.
>
> **Nota:** `inversion_usd` no aparece en este `describe()` porque todavía es texto (`object`) — pandas la ignora. La vamos a ver recién después de A.2, cuando la convirtamos a numérica.

### Celda 4 — Diagnóstico A.1: boxplot de `ventas_usd`

```python
plt.figure(figsize=(7, 2.5))
sns.boxplot(x=df["ventas_usd"])
plt.title("Ventas — vista cruda (¿hay outliers?)")
plt.show()
```

| Bloque | Efecto |
|---|---|
| `plt.figure(figsize=(7, 2.5))` | Crea una figura nueva con tamaño 7×2.5 pulgadas. |
| `sns.boxplot(x=df["ventas_usd"])` | Boxplot horizontal de la columna `ventas_usd`. La caja muestra el rango intercuartílico (Q1-Q3), la línea central es la mediana, y los puntos sueltos son outliers según la regla del IQR. |
| `plt.show()` | Renderiza el gráfico. |

> **Observación esperada:** el boxplot muestra un punto muy alejado del resto → confirma el outlier extremo en `ventas_usd` que ya habíamos visto en el `describe()`.

### Celda 5 — Diagnóstico A.1: formato de `inversion_usd`

```python
# Formato de inversion_usd: viene como texto con $ y comas
print("Tipo de inversion_usd:", df["inversion_usd"].dtype)
print(df["inversion_usd"].head(6).tolist())
```

| Bloque | Efecto |
|---|---|
| `df["inversion_usd"].dtype` | Devuelve el tipo de dato de la columna (`object`, `float64`, etc.). Un `object` en una columna que debería ser numérica = problema. |
| `.head(6).tolist()` | Toma los primeros 6 valores y los convierte a una lista de Python — más legible que un Series para inspeccionar visualmente strings. |

> **Observación esperada:** la columna llegó como `object` con valores tipo `"$1,209"`, `"$2,174.78"`, `"USD 980"` — formatos inconsistentes mezclando símbolo, separador de miles y prefijo. Esto motiva A.2.

---

### Celda 6 — A.2: arreglar el formato de `inversion_usd`

```python
df["inversion_usd"] = (
    df["inversion_usd"]
    .str.replace("$", "", regex=False)
    .str.replace("USD", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["inversion_usd"] = pd.to_numeric(df["inversion_usd"], errors="coerce")
```

| Bloque | Efecto |
|---|---|
| `.str.replace("$", "", regex=False)` | Saca todos los `$` de los valores. `regex=False` evita que pandas trate `$` como un metacaracter de expresión regular (importante: `$` significa "fin de string" en regex). |
| `.str.replace("USD", "", ...)` | Saca el prefijo `USD ` de los valores que lo tengan. |
| `.str.replace(",", "", ...)` | Saca las comas (separadores de miles). |
| `.str.strip()` | Saca los espacios al principio y al final del string. |
| `pd.to_numeric(..., errors="coerce")` | Convierte el string a número. `errors="coerce"` significa: si algo no se puede convertir (por ejemplo un valor que era nulo originalmente), poné `NaN` en lugar de tirar un error. |

> **Por qué hay que formatear `inversion_usd`**: la columna llegó como `object` (texto) porque el export del CRM serializó los montos para mostrar (`"$1,209"`, `"USD 980"`). Mientras siga siendo texto, **no podemos**: (a) calcular la mediana para imputar nulos, (b) calcular correlaciones, (c) usarla como predictor en `LinearRegression`. Pandas trataría `"$1,209"` y `"$1209"` como dos categorías distintas, y cualquier operación aritmética tira error. Es un caso clásico de **export de sistema legacy** que mezcla datos con formato de presentación.
>
> **Por qué este paso va primero (antes que la imputación)**: las imputaciones numéricas (`fillna(median())`) requieren que la columna sea numérica. Si dejáramos esto para después, `df["inversion_usd"].median()` fallaría con un `TypeError`. Hay un orden lógico en la limpieza: primero el tipo, después los nulos, después los outliers.

---

### Celda 7 — A.3: imputar nulos con la mediana

```python
for col in ["inversion_usd", "clicks"]:
    mediana = df[col].median()
    df[col] = df[col].fillna(mediana)
    print(f"{col}: imputado con mediana = {mediana:.2f}")

print("\nNulos restantes:")
print(df.isnull().sum())
```

| Bloque | Efecto |
|---|---|
| `for col in [...]` | Itera sobre las dos columnas que tienen nulos. Hacer un loop evita repetir código. |
| `df[col].median()` | Calcula la mediana de la columna (ignora los nulos automáticamente). |
| `df[col].fillna(mediana)` | Reemplaza los `NaN` con el valor de la mediana. |
| `f"...{mediana:.2f}"` | f-string con formato: muestra la mediana con 2 decimales. |

> **Por qué imputamos (en lugar de eliminar las filas con nulos)**: tenemos 93 filas en total. Eliminar 9 filas (5 + 4) sería tirar ~10% del dataset por culpa de pocos faltantes. En un dataset chico como este, **cada fila cuenta**. Imputar nos permite conservar la fila completa para todas las otras variables que sí están bien.
>
> **Por qué mediana y no media**: la mediana es **robusta a outliers**. Como ya sabemos que hay un outlier importante en `ventas_usd` (~$45.000), usar la media inflaría artificialmente las imputaciones — la media de una columna con un valor extremo se "tira" hacia ese valor. La mediana, por construcción, ignora la magnitud de los extremos: solo le importa el orden de los datos. Refleja el valor "típico" del día normal de campaña, que es lo que queremos cuando no sabemos qué pasó realmente ese día.
>
> **Por qué no usamos la moda o un valor fijo (ej. 0)**: la moda solo tiene sentido para variables categóricas. Imputar con 0 sería falso — un día con `inversion_usd = NaN` no es lo mismo que "ese día no se invirtió"; es "no sabemos cuánto se invirtió". Poner 0 metería un error sistemático en el modelo.

---

### Celda 8 — A.4: normalizar categorías de `canal` y `region`

```python
df["canal"] = df["canal"].str.lower().str.strip()
df["region"] = df["region"].str.lower().str.strip()

mapa_canal = {
    "instagram": "Instagram", "ig": "Instagram",
    "google ads": "Google Ads", "google_ads": "Google Ads", "googleads": "Google Ads",
    "tiktok": "TikTok", "tik tok": "TikTok",
    "facebook": "Facebook", "fb": "Facebook",
}
df["canal"] = df["canal"].map(mapa_canal)

mapa_region = {
    "caba": "CABA", "capital": "CABA",
    "gba": "GBA",
    "interior": "Interior",
}
df["region"] = df["region"].map(mapa_region)
```

| Bloque | Efecto |
|---|---|
| `.str.lower()` | Pasa todo el string a minúscula. Después de esta línea, `Instagram` y `INSTAGRAM` son iguales. |
| `.str.strip()` | Saca los espacios al principio y al final (ej: `"ig "` → `"ig"`). |
| `mapa_canal = {...}` | Diccionario que define el "nombre canónico" de cada variante. La clave es la variante en minúscula+strip; el valor es el nombre limpio que queremos. |
| `df["canal"].map(mapa_canal)` | Reemplaza cada valor por lo que el diccionario diga. **Si un valor no está en el diccionario, queda como `NaN` — por eso es importante incluir todas las variantes.** |

> **Por qué este patrón es mejor que `.replace`**: `.map()` con un diccionario es explícito y fácil de auditar. Si después aparece un nuevo canal en los datos, salta como nulo y lo detectás de inmediato.

---

### Celda 9 — A.5: detectar y eliminar outliers con IQR

```python
Q1 = df["ventas_usd"].quantile(0.25)
Q3 = df["ventas_usd"].quantile(0.75)
IQR = Q3 - Q1
lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
print(f"Q1={Q1:.0f}  Q3={Q3:.0f}  IQR={IQR:.0f}")
print(f"Límites aceptables: [{lim_inf:.0f}, {lim_sup:.0f}]")

n_antes = len(df)
df = df[(df["ventas_usd"] >= lim_inf) & (df["ventas_usd"] <= lim_sup)].copy()
print(f"Filas antes: {n_antes}  →  después: {len(df)}  ({n_antes - len(df)} outlier eliminado)")

plt.figure(figsize=(7, 2.5))
sns.boxplot(x=df["ventas_usd"])
plt.title("Ventas — después de quitar outliers")
plt.show()
```

| Bloque | Efecto |
|---|---|
| `df["ventas_usd"].quantile(0.25)` | Calcula el primer cuartil (Q1) — el valor por debajo del cual está el 25% de los datos. |
| `quantile(0.75)` | Tercer cuartil (Q3) — el valor por debajo del cual está el 75% de los datos. |
| `IQR = Q3 - Q1` | Rango intercuartílico — el ancho de la "caja central" del boxplot. |
| `lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR` | Asignación múltiple. Define los límites aceptables según la regla clásica de Tukey: cualquier valor fuera de este rango es un outlier candidato. |
| `df[(...) & (...)]` | Filtra el DataFrame manteniendo solo las filas cuyo `ventas_usd` está dentro de los límites. El `&` es AND lógico (cada condición tiene que ir entre paréntesis). |
| `.copy()` | Crea una copia explícita para evitar el `SettingWithCopyWarning` de pandas. |
| Boxplot final | Visualiza cómo quedó la distribución después de quitar el outlier. |

> **Por qué usamos la regla del IQR (y no z-score, desviación estándar, etc.)**:
>
> 1. **El IQR es robusto** — usa cuartiles, que no se ven afectados por los valores extremos en sí. Es una paradoja del z-score: para detectar outliers calcula media y desvío estándar, pero **esos dos estadísticos están justamente influenciados por los outliers que queremos detectar**. Resultado: el z-score "esconde" outliers en datasets contaminados. El IQR no tiene ese problema.
> 2. **No asume distribución normal**. El z-score asume que los datos vienen de una campana de Gauss; si la distribución es asimétrica (como suele pasar con ventas, precios, ingresos — todas tienen cola larga a la derecha), el z-score marca como "outliers" valores que son perfectamente normales para esa distribución. El IQR no asume nada sobre la forma de la distribución.
> 3. **Es coherente con el boxplot**. La regla `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` es exactamente la que usan los "bigotes" del boxplot. Cuando ves un punto fuera del boxplot, ese punto es un outlier por IQR. Esto hace que el diagnóstico visual y el filtro programático coincidan — más fácil de auditar y comunicar.
> 4. **El factor 1.5 es la convención de Tukey**, balance estándar entre falsos positivos y falsos negativos. Para outliers más extremos se usa 3.0 ("far outliers").
>
> **Decisión de negocio (eliminar vs. conservar)**: eliminamos el outlier porque a simple vista es un valor de ~$45.000 que claramente es un error de carga (el segundo más alto del dataset es ~$8.500 — no hay forma de que sea una transición suave). En otros contextos, un outlier puede ser una señal real (una campaña viral, una venta excepcional, un día de Black Friday) y conviene conservarlo o analizarlo aparte. **La técnica (IQR) detecta candidatos; la decisión final siempre es de negocio.**

---

### Celda 10 — A.6: eliminar duplicados

```python
print(f"Duplicados antes: {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicados después: {df.duplicated().sum()}")
print(f"Filas finales: {len(df)}")
```

| Bloque | Efecto |
|---|---|
| `df.duplicated()` | Devuelve una Serie de True/False: True para cada fila que es **duplicado exacto** de una fila anterior. La primera ocurrencia siempre es False. |
| `.sum()` | Cuenta cuántas filas son duplicadas. |
| `df.drop_duplicates()` | Devuelve un nuevo DataFrame sin filas duplicadas. Por defecto conserva la primera ocurrencia (`keep="first"`). |
| `.reset_index(drop=True)` | Reinicia el índice numérico desde 0. Si no se hace, el índice queda con "huecos" donde estaban los duplicados. |

---

### Celda 11 — A.7: verificación final y `df_clean`

```python
df_clean = df.copy()
print(f"df_clean listo: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas")
print(f"Nulos totales: {df_clean.isnull().sum().sum()}")
df_clean.head()
```

| Bloque | Efecto |
|---|---|
| `df_clean = df.copy()` | Crea una copia del DataFrame limpio en una variable nueva. **Convención**: a partir de acá, todo el código de la Parte B usa `df_clean` en lugar de `df`. Esto deja claro que estamos trabajando con datos validados. |
| `df_clean.isnull().sum().sum()` | Doble `.sum()`: el primero suma por columna, el segundo suma todo. Devuelve un solo número: cuántos nulos quedan en total en todo el DataFrame. Esperamos `0`. |

---

## Parte B — Regresión Lineal

### Celda 12 — B.1: estadísticas descriptivas (repaso post-cleaning)

```python
df_clean.describe()
```

`.describe()` devuelve para cada columna numérica: count, mean, std, min, 25%, 50% (mediana), 75%, max. Es la primera lectura cuantitativa del dataset.

### Celda 13 — B.1: heatmap de correlación

```python
numericas = df_clean.select_dtypes(include="number")
plt.figure(figsize=(6, 4))
sns.heatmap(numericas.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlaciones entre variables numéricas")
plt.show()
```

| Bloque | Efecto |
|---|---|
| `df_clean.select_dtypes(include="number")` | Selecciona solo las columnas numéricas (descarta `fecha`, `canal`, `region`). |
| `.corr()` | Calcula la matriz de correlación de Pearson entre todas las pares de columnas numéricas. Devuelve una tabla cuadrada con valores entre -1 (correlación negativa perfecta) y 1 (positiva perfecta). |
| `sns.heatmap(...)` | Renderiza la matriz como un mapa de calor. |
| `annot=True` | Escribe el valor numérico de la correlación dentro de cada celda. |
| `cmap="coolwarm"` | Paleta divergente: azul para correlaciones negativas, rojo para positivas, blanco para 0. |
| `fmt=".2f"` | Formato del número dentro de cada celda: 2 decimales. |
| `vmin=-1, vmax=1` | Fija la escala de colores entre -1 y 1, no en el rango observado de los datos. Esto hace que dos heatmaps de datasets distintos sean comparables. |

> **Lectura del output**: la correlación más fuerte con `ventas_usd` debería ser `inversion_usd` (~0.97). Eso valida nuestra elección de predictor.

---

### Celda 14 — B.2: definir `X` (predictor) e `y` (target)

```python
X = df_clean[["inversion_usd"]]
y = df_clean["ventas_usd"]
```

| Bloque | Efecto |
|---|---|
| `df_clean[["inversion_usd"]]` | Doble corchete → devuelve un **DataFrame** (2D). |
| `df_clean["ventas_usd"]` | Corchete simple → devuelve una **Series** (1D). |

> **Por qué `X` necesita doble corchete**: scikit-learn espera que `X` sea una "matriz de features" — una estructura 2D, aunque tenga una sola columna. `y` puede ser 1D porque es el target. **Es la convención más común que confunde a los principiantes.**

---

### Celda 15 — B.3: train/test split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} filas  |  Test: {len(X_test)} filas")
```

| Bloque | Efecto |
|---|---|
| `train_test_split(X, y, ...)` | Divide los datos en dos pares: 80% train + 20% test (porque `test_size=0.2`). Devuelve cuatro objetos en este orden: `X_train`, `X_test`, `y_train`, `y_test`. |
| `random_state=42` | Fija la semilla del generador aleatorio. **Si lo corrés mañana, te da el mismo split exacto.** Es esencial para reproducibilidad — sin esto, cada vez que corre el notebook el split sería distinto y los resultados también. |

> **Por qué separamos**: el modelo se entrena con `train` y se evalúa con `test` — datos que **nunca vio**. Esto evita engañarnos con un modelo que solo memorizó los ejemplos (overfitting — ver "Para ver en casa" en el notebook).

---

### Celda 16 — B.4: entrenar el modelo y predecir

```python
modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
```

| Bloque | Efecto |
|---|---|
| `LinearRegression()` | Instancia un objeto de regresión lineal. **Sin hiperparámetros** — la regresión lineal simple no tiene nada que tunear. |
| `modelo.fit(X_train, y_train)` | "Entrena" el modelo: encuentra los coeficientes (`m`, `b`) que minimizan el error cuadrático en el set de entrenamiento. |
| `modelo.predict(X_test)` | Aplica la fórmula `y = m·x + b` a cada fila del set de test y devuelve las predicciones. |

> **El patrón `fit` + `predict`** es universal en scikit-learn: lo vas a ver en cualquier modelo (regresión logística, árboles, random forest, etc.). Cambia el modelo, el patrón es el mismo.

---

### Celda 17 — B.5: métricas de evaluación

```python
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}  (en USD)")
print(f"R²   : {r2:.3f}")
```

| Métrica | Qué mide | Unidad | Cómo se lee |
|---|---|---|---|
| **MSE** (Mean Squared Error) | Promedio de los errores al cuadrado: para cada fila del test, `(predicho − real)²`, y se promedia. Elevar al cuadrado evita que los errores positivos y negativos se cancelen, y penaliza más fuerte los errores grandes. | USD² | **No es interpretable directamente** — nadie habla en "dólares al cuadrado". Lo mostramos porque es la métrica que el modelo **minimiza por dentro** cuando entrena (la regresión lineal busca los `m` y `b` que hacen el MSE más chico posible) y porque el RMSE es literalmente su raíz cuadrada. |
| **RMSE** (Root MSE) | Raíz del MSE → vuelve a la unidad original. | USD (la misma que `y`) | "El modelo se equivoca, en promedio, en ±$RMSE al predecir las ventas de un día." Es **la métrica que se comunica a negocio**. |
| **R²** (coef. de determinación) | Compara el modelo contra un "baseline tonto" que siempre predice el promedio de `y`. Mide qué proporción de la variabilidad del target captura el modelo. | Sin unidad (0 a 1) | `R² = 0` → no es mejor que predecir el promedio. `R² = 1` → predicción perfecta. `R² = 0.95` → captura el 95% de la variabilidad. **Puede ser negativo** si el modelo es peor que el baseline. |

> **MSE vs RMSE — la regla mental**: MSE es el número que el modelo optimiza. RMSE es el mismo número, pero traducido a una unidad que entiende un humano. Por eso siempre se reportan juntos pero el que se interpreta es el RMSE.
>
> **Cuándo un RMSE es "bueno"**: depende del orden de magnitud de `y`. RMSE solo no se interpreta — siempre se compara contra la magnitud típica de la variable predicha. RMSE de $356 sobre ventas medianas de ~$4.700 = ~7% → bueno. El mismo RMSE sobre ventas medianas de $400 sería pésimo.
>
> **Cuándo un R² es "bueno"**: depende del dominio. En negocio digital, >0.6 ya es decente, 0.7-0.8 es muy bueno. **Un R² de 0.95 es atípico** — este dataset está armado para que la relación sea limpia. Si aparece con datos reales, sospechar *data leakage* (la `y` se metió en la `X`).
>
> **Por qué RMSE y no MAE**: MAE (Mean Absolute Error) promedia errores en valor absoluto. RMSE penaliza más los errores grandes por el cuadrado. En negocio normalmente preferís RMSE: equivocarte "feo" un día es peor que equivocarte poco varios días.
>
> **Por qué necesitamos R² *y* RMSE — responden preguntas distintas**:
> - **RMSE** responde "¿por cuánto se equivoca mi modelo?" → en dólares, para el CEO.
> - **R²** responde "¿qué proporción de la señal capturé?" → sin unidad, para comparar modelos.
> - Las dos juntas son la evaluación completa: "el modelo explica el 95% de la variabilidad y cuando se equivoca lo hace por ~$356".

---

### Celda 18 — B.6: extraer e interpretar los coeficientes

```python
m = modelo.coef_[0]
b = modelo.intercept_
print(f"Pendiente (m): {m:.2f}")
print(f"Intercepto (b): {b:.2f}")
print()
print("Lectura de negocio:")
print(f"-> Por cada USD adicional invertido en marketing digital,")
print(f"   las ventas aumentan en promedio ${m:.2f}.")
print(f"-> Con inversión cero, el modelo estima ventas base de ${b:.2f}.")
```

| Bloque | Efecto |
|---|---|
| `modelo.coef_` | Atributo del modelo entrenado: array con los coeficientes (uno por feature). En regresión simple hay uno solo, por eso `[0]`. |
| `modelo.intercept_` | El término independiente (intercepto / `b`). |

> **Cómo interpretar `m`**: si `m = 3.10`, significa que **por cada peso adicional invertido, las ventas suben en promedio $3.10**. Esa es la "respuesta de negocio".
>
> **Cómo interpretar `b` (con cuidado)**: el intercepto es la predicción del modelo cuando `inversion = 0`. Si da `$543`, significa que el modelo estima ~$543 de ventas base sin inversión. Hay que tomarlo con pinzas — el modelo no fue entrenado con valores cercanos a 0, así que es una **extrapolación**, no una predicción confiable.

---

### Celda 19 — B.7: visualización scatter + recta de regresión

```python
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color="steelblue", label="Datos reales (test)", alpha=0.7)

x_linea = np.linspace(X["inversion_usd"].min(), X["inversion_usd"].max(), 100).reshape(-1, 1)
plt.plot(x_linea, modelo.predict(x_linea), color="red", linewidth=2, label="Modelo lineal")

plt.title("Regresión lineal: Inversión vs Ventas")
plt.xlabel("Inversión (USD)")
plt.ylabel("Ventas (USD)")
plt.legend()
plt.show()
```

| Bloque | Efecto |
|---|---|
| `plt.scatter(X_test, y_test, ...)` | Dibuja los puntos del set de test. `alpha=0.7` los hace semitransparentes para que se vean superposiciones. |
| `np.linspace(min, max, 100)` | Genera 100 puntos equiespaciados entre el mínimo y el máximo de `inversion_usd`. **No usamos los `X` reales** porque no están ordenados — `linspace` da una secuencia limpia para dibujar la línea. |
| `.reshape(-1, 1)` | Convierte el array de 1D (`(100,)`) a 2D (`(100, 1)`). Es necesario porque `modelo.predict()` espera la misma forma 2D que tenía `X_train`. |
| `modelo.predict(x_linea)` | Calcula la `y` predicha para cada uno de los 100 puntos de `x_linea`. |
| `plt.plot(x_linea, ...)` | Dibuja la recta del modelo encima del scatter. |
| `plt.legend()` | Muestra la leyenda con los `label` que pusimos en cada `plt.scatter` / `plt.plot`. |

> **Lectura de la imagen**: si los puntos están cerca de la recta → el modelo predice bien. Si están dispersos → el modelo no captura toda la variabilidad.

---

## Recursos relacionados

- **`Lineal_Simple_Regression.ipynb`** (en esta misma carpeta): notebook de ejemplo con el mismo flujo de regresión lineal aplicado a un dataset de salarios vs años de experiencia. Incluye un paso adicional con `statsmodels` para extraer p-values y F-statistic — útil si querés ver cómo se valida la significancia estadística de los coeficientes.
- **`Clase 4 - Data Cleaning & Linear Regression.pdf`**: las slides de la clase teórica de Pablo, con la cobertura conceptual completa.
- **`hallazgos.md`**: el análisis de qué encontramos en los datos y qué decisiones tomamos en clave de negocio.
