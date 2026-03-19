# Guía de Explicaciones — startups_practica.ipynb
## Analítica de Datos · Universidad de San Andrés

---

## CELDA 03 — Instalación de librerías

```python
!pip install pandas matplotlib seaborn --quiet
```

El `!` al principio significa que esta línea no es Python — es un comando de terminal que se ejecuta desde el notebook. `pip install` descarga e instala las tres librerías. `--quiet` suprime los mensajes de instalación para que la salida sea más limpia.

- **pandas** — manejo de datos en tablas (dataframes)
- **matplotlib** — graficación base
- **seaborn** — graficación de alto nivel, construida sobre matplotlib

---

## CELDA 04 — Imports y configuración

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
```

`import X as Y` carga la librería X y le asigna el alias Y. Son convenciones universales — en todo el mundo se usan `pd`, `plt` y `sns`.

`sns.set_style('whitegrid')` aplica el estilo visual de fondo blanco con grilla gris a todos los gráficos que se hagan después. Solo hay que ejecutarla una vez.

---

## CELDA 06 — Cargar el dataset

```python
df_startups = pd.read_csv('startups.csv', index_col=0)
```

`pd.read_csv()` lee el archivo CSV y lo convierte en un DataFrame — la estructura de tabla de pandas. `index_col=0` le dice que la primera columna del CSV es el índice (número de fila), no una columna de datos. El resultado se guarda en la variable `df_startups`.

---

## CELDA 07 — Primeras filas

```python
df_startups.head()
```

Muestra las primeras 5 filas del DataFrame. Sin argumento muestra 5 por defecto. `head(10)` mostraría 10.

---

## CELDA 08 — Últimas filas

```python
df_startups.tail()
```

Igual que `head()` pero desde el final. Útil para verificar que el dataset no esté cortado o que los últimos registros sean coherentes.

---

## CELDA 09 — Dimensiones

```python
print(f"El dataset contiene {df_startups.shape[0]} filas y {df_startups.shape[1]} columnas")
```

`shape` devuelve una tupla con (filas, columnas). `shape[0]` es el número de filas, `shape[1]` es el número de columnas. La `f` antes del string permite insertar variables dentro de `{}` directamente.

---

## CELDA 10 — Nombres de columnas

```python
df_startups.columns.tolist()
```

`columns` devuelve el índice de columnas. `.tolist()` lo convierte en una lista de Python normal, más fácil de leer.

---

## CELDA 12 — Tipos de dato

```python
df_startups.info()
```

Muestra el tipo de dato de cada columna (`int64`, `float64`, `object`) y cuántos valores no son nulos. Es el primer chequeo de salud del dataset. Si una columna numérica aparece como `object`, hay un problema (letras mezcladas con números, por ejemplo).

---

## CELDA 13 — Estadísticas descriptivas

```python
df_startups.describe().round(2)
```

Genera automáticamente para cada columna numérica: cantidad de valores, media, desvío estándar, mínimo, cuartiles (25%, 50%, 75%) y máximo. `.round(2)` redondea a 2 decimales. La fila `50%` es la mediana.

---

## CELDA 15 — Comparar media vs mediana

```python
cols = ['monto_ronda_usd_mm', 'valuacion_usd_mm', 'empleados']

for col in cols:
    media   = df_startups[col].mean()
    mediana = df_startups[col].median()
    ratio   = media / mediana
    print(f"{col}")
    print(f"  Media:   {media:>10,.0f}")
    print(f"  Mediana: {mediana:>10,.0f}")
    print(f"  Ratio:   {ratio:>10.1f}x  <- cuanto mas alto, mas sesgado")
    print()
```

`cols` es una lista con los nombres de las columnas a analizar. El `for` recorre cada columna y calcula media, mediana y el ratio entre ambas. El ratio indica cuánto está distorsionado el promedio — si da 3x significa que la media es el triple de la mediana, lo que indica que hay outliers jalando el promedio hacia arriba.

El formato `{media:>10,.0f}` alinea a la derecha (`:>`), usa separador de miles (`,`) y no muestra decimales (`.0f`).

---

## CELDA 16 — Top 3 por columna

```python
for col in cols:
    print(f"Top 3 por {col}:")
    print(df_startups.nlargest(3, col)[['startup', col]].to_string(index=False))
    print()
```

Usa la misma lista `cols` del bloque anterior. `.nlargest(3, col)` devuelve las 3 filas con el valor más alto en esa columna. `to_string(index=False)` imprime la tabla sin el índice numérico de la izquierda, más limpio para imprimir.

---

## CELDA 17 — value_counts

```python
df_startups['startup'].value_counts()
```

Cuenta cuántas veces aparece cada valor, ordenado de mayor a menor. Si una startup aparece más de una vez significa que hay duplicados.

---

## CELDA 18 — nunique

```python
df_startups['startup'].nunique()
```

Devuelve un número: cuántos valores distintos hay. `nunique` = number of unique values.

---

## CELDA 19 — unique

```python
df_startups['startup'].unique()
```

Devuelve la lista de valores distintos. A diferencia de `nunique()` que devuelve el conteo, `unique()` devuelve los valores en sí.

---

## CELDA 21 — Contar nulos

```python
df_startups.isnull().sum()
```

`isnull()` genera una tabla de True/False — True donde hay nulo. `.sum()` suma los True de cada columna (True = 1, False = 0). El resultado es el total de nulos por columna.

---

## CELDA 22 — Nulos en porcentaje

```python
(df_startups.isnull().sum() / len(df_startups) * 100).round(1)
```

Mismo cálculo que antes pero dividido por el total de filas (`len`) y multiplicado por 100. `.round(1)` redondea a un decimal.

---

## CELDA 23 — Tabla de nulos

```python
nulos = df_startups.isnull().sum()
porcentaje = (nulos / len(df_startups) * 100).round(1)

tabla_nulos = pd.DataFrame({
    'nulos': nulos,
    '%': porcentaje
})

tabla_nulos[tabla_nulos['nulos'] > 0].sort_values('nulos', ascending=False)
```

Combina el conteo y el porcentaje en una tabla más legible. `pd.DataFrame({...})` crea un DataFrame a partir de un diccionario — las claves son los nombres de columna, los valores son las Series. Al final filtra solo las columnas con al menos un nulo y las ordena de mayor a menor.

---

## CELDA 25 — Ver filas con nulos

```python
df_startups[df_startups['valuacion_usd_mm'].isnull()][['startup', 'pais', 'sector', 'ultima_ronda','valuacion_usd_mm']].head()
```

Filtra las filas donde `valuacion_usd_mm` es nulo y muestra solo algunas columnas. Permite identificar qué empresas específicas no tienen valuación y entender por qué — en este caso son empresas autofinanciadas o que no divulgan su valuación públicamente.

---

## CELDA 26 — Contar duplicados

```python
df_startups.duplicated().sum()
```

`duplicated()` devuelve True para cada fila que es un duplicado exacto de una anterior. `.sum()` cuenta cuántas hay.

---

## CELDA 27 — Ver duplicados

```python
df_startups[df_startups.duplicated(keep=False)].sort_values('startup').head()
```

`keep=False` marca TODAS las ocurrencias de filas duplicadas como True, no solo las repetidas. Así se ven tanto el original como el duplicado. `.sort_values('startup')` las ordena alfabéticamente para que sea fácil ver los pares.

---

## CELDA 28 — Eliminar duplicados

```python
df_clean = df_startups.drop_duplicates(keep='first').reset_index(drop=True)

print(f"Antes : {len(df_startups)} filas")
print(f"Despues: {len(df_clean)} filas")
```

`drop_duplicates(keep='first')` elimina duplicados conservando la primera ocurrencia de cada fila. El resultado se guarda en `df_clean` — una nueva variable, sin modificar `df_startups`. `reset_index(drop=True)` reinicia la numeración del índice desde 0. A partir de esta celda se trabaja siempre con `df_clean`.

---

## CELDA 32 — Seleccionar una columna

```python
df_clean['startup'].head(10)
```

`df['columna']` selecciona una columna y devuelve una Serie (como una lista con índice). `.head(10)` muestra los primeros 10 valores.

---

## CELDA 33 — Seleccionar múltiples columnas

```python
df_clean[['startup', 'sector', 'valuacion_usd_mm']].head(10)
```

`df[['col1', 'col2']]` con doble corchete selecciona múltiples columnas y devuelve un DataFrame. Un corchete = Serie, dos corchetes = DataFrame.

---

## CELDA 34 — Filtrar por texto

```python
df_usa = df_clean[df_clean['pais'] == 'USA']
print(f"Startups de USA: {len(df_usa)}")
df_usa[['startup', 'sector', 'valuacion_usd_mm']].head(8)
```

`df_clean['pais'] == 'USA'` genera una Serie de True/False. Al pasarla como índice `df_clean[...]` filtra solo las filas donde es True. El resultado se guarda en `df_usa`. Luego se imprime cuántas hay y se muestran las primeras 8.

---

## CELDA 35 — Filtrar por número

```python
df_clean[df_clean['valuacion_usd_mm'] > 10000][['startup', 'pais', 'valuacion_usd_mm']].sort_values('valuacion_usd_mm', ascending=False)
```

Igual que el filtro anterior pero con condición numérica: solo las filas donde la valuación supera USD 10.000M. `.sort_values(..., ascending=False)` ordena de mayor a menor.

---

## CELDA 36 — Filtro con dos condiciones

```python
df_clean[(df_clean['pais'] == 'USA') & (df_clean['valuacion_usd_mm'].notna())][['startup', 'sector', 'valuacion_usd_mm']].head(8)
```

Combina dos condiciones con `&` (AND) — solo pasan las filas donde las DOS son True. `.notna()` es el opuesto de `.isnull()`: devuelve True donde hay un valor, False donde hay nulo. Cada condición va entre paréntesis cuando se combinan.

---

## CELDA 40 — Filtrar con lista (isin)

```python
paises_top = ['USA', 'UK', 'Israel', 'China', 'Canada']
df_clean[df_clean['pais'].isin(paises_top)][['startup', 'pais', 'sector']]
```

`.isin()` es la forma elegante de hacer varios `==` con `|`. En vez de escribir `pais == 'USA' | pais == 'UK' | ...`, se pasa una lista. Devuelve True para cada fila donde el valor está en la lista.

---

## CELDA 42 — Ordenar por valuación

```python
df_clean.sort_values('valuacion_usd_mm', ascending=False).head(10)[['startup', 'pais', 'sector', 'valuacion_usd_mm']]
```

`.sort_values()` ordena el DataFrame por la columna indicada. `ascending=False` = de mayor a menor. `.head(10)` toma los primeros 10. La selección de columnas al final muestra solo las relevantes.

---

## CELDA 43 — Ordenar por empleados

```python
df_clean.sort_values('empleados', ascending=False).head(10)[['startup', 'sector', 'empleados', 'valuacion_usd_mm']]
```

Mismo patrón — ordena por empleados de mayor a menor y muestra el top 10.

---

## CELDA 45 — Nueva columna: edad

```python
df_clean['edad_anios'] = 2025 - df_clean['anio_fundacion']
df_clean[['startup', 'anio_fundacion', 'edad_anios']].head(8)
```

`df_clean['nombre_nueva_columna'] = operacion` crea una nueva columna. La operación se aplica fila por fila automáticamente — no hace falta un loop. `2025 - anio_fundacion` calcula cuántos años tiene cada startup.

---

## CELDA 46 — Nueva columna: es_unicornio

```python
df_clean['es_unicornio'] = df_clean['valuacion_usd_mm'] > 1000

df_clean[['startup', 'valuacion_usd_mm', 'es_unicornio']].head(10)
```

Una condición (`> 1000`) devuelve True o False para cada fila, y eso se guarda directamente como una nueva columna booleana. Un "unicornio" en el mundo startup es una empresa privada con valuación mayor a USD 1.000M.

---

## CELDA 47 — Contar unicornios

```python
df_clean['es_unicornio'].value_counts()
```

`value_counts()` en una columna booleana cuenta cuántos True y cuántos False hay. Dice directamente cuántos unicornios hay en el dataset.

---

## CELDA 49 — Groupby simple

```python
df_clean.groupby('sector')['valuacion_usd_mm'].mean().round(1).sort_values(ascending=False).reset_index(drop=False).rename(columns={'valuacion_usd_mm': 'valuacion_promedio_usd_mm'})
```

Se lee en pasos:
1. `groupby('sector')` — agrupa las filas por sector
2. `['valuacion_usd_mm']` — selecciona esa columna dentro de cada grupo
3. `.mean()` — calcula el promedio de cada grupo
4. `.round(1)` — redondea
5. `.sort_values(ascending=False)` — ordena de mayor a menor
6. `.reset_index()` — convierte el sector de índice a columna normal
7. `.rename(...)` — cambia el nombre de la columna a algo más descriptivo

---

## CELDA 50 — Groupby con agg

```python
resumen_sector = df_clean.groupby('sector').agg(
    cantidad     = ('startup',         'count'),
    valuacion_md = ('valuacion_usd_mm', 'median'),
    monto_md     = ('monto_ronda_usd_mm','median'),
    empleados_md = ('empleados',        'median'),
).round(1).sort_values('cantidad', ascending=False)
```

`.agg()` permite calcular múltiples métricas a la vez. Cada línea dentro define una columna nueva: `nombre_columna = ('columna_original', 'función')`. Las funciones disponibles son `count`, `mean`, `median`, `max`, `min`, `sum`. La mediana es más robusta que la media para datos con outliers.

---

## CELDA 51 — Groupby por país

```python
tabla = (df_clean.groupby('pais')['startup']
         .count()
         .sort_values(ascending=False)
         .reset_index()
         .rename(columns={'startup': 'cantidad'}))

tabla
```

Mismo patrón pero escrito en múltiples líneas con paréntesis para mayor legibilidad. Cuenta cuántas startups hay por país y lo presenta como tabla ordenada.

---

## CELDA 52 — Groupby por etapa de ronda

```python
df_clean.groupby('ultima_ronda').agg(
    cantidad     = ('startup',          'count'),
    valuacion_md = ('valuacion_usd_mm',  'median'),
    monto_md     = ('monto_ronda_usd_mm','median'),
).round(1)
```

Mismo patrón que la celda 50 pero agrupado por etapa de ronda. Permite ver cómo la valuación y el monto de ronda crecen a medida que la startup avanza en etapas.

---

## CELDA 57 — Countplot por sector

```python
orden = df_clean['sector'].value_counts().index

plt.figure(figsize=(11, 6))
sns.countplot(data=df_clean, y='sector', order=orden, palette='tab20')
plt.title('Startups de IA por sector')
plt.xlabel('Cantidad')
plt.ylabel('')
plt.tight_layout()
plt.show()
```

`value_counts().index` extrae el orden de sectores de mayor a menor. `plt.figure(figsize=(11,6))` define el tamaño del gráfico en pulgadas. `y='sector'` hace el countplot horizontal — útil cuando los labels son largos. `order=orden` fuerza el orden de mayor a menor. `palette='tab20'` usa 20 colores distintos. `plt.tight_layout()` ajusta los márgenes para que nada se corte. `plt.show()` muestra el gráfico.

---

## CELDA 58 — Porcentaje por sector

```python
df_clean['sector'].value_counts(normalize=True).round(2) * 100
```

`normalize=True` devuelve proporciones en lugar de conteos (entre 0 y 1). Multiplicado por 100 da el porcentaje de cada sector sobre el total.

---

## CELDA 59 — Countplot por país

```python
orden_paises = df_clean['pais'].value_counts().head(10).index

plt.figure(figsize=(10, 5))
sns.countplot(data=df_clean[df_clean['pais'].isin(orden_paises)],
              y='pais', order=orden_paises, palette='tab20')
plt.title('Top 10 paises por cantidad de startups')
plt.xlabel('Cantidad')
plt.ylabel('')
plt.tight_layout()
plt.show()
```

`.head(10).index` toma solo los 10 países con más startups. El filtro `df_clean[df_clean['pais'].isin(orden_paises)]` asegura que el gráfico solo incluya esos 10 países — sin esto seaborn incluiría todos aunque `order` los ordene.

---

## CELDA 60 — Porcentaje por país

```python
df_clean['pais'].value_counts(normalize=True).round(2) * 100
```

Igual que la celda 58 pero para países. Permite ver que USA representa el 71% del dataset.

---

## CELDA 63 — Histograma de monto de ronda

```python
media   = df_clean['monto_ronda_usd_mm'].mean()
mediana = df_clean['monto_ronda_usd_mm'].median()

plt.figure(figsize=(11, 5))
sns.histplot(data=df_clean, x='monto_ronda_usd_mm', bins=25, kde=True)
plt.axvline(media,   color='red',   linestyle='--', label=f'Media:   USD {media:,.0f}M')
plt.axvline(mediana, color='black', linestyle='--', label=f'Mediana: USD {mediana:,.0f}M')
plt.title('Distribucion del monto de ronda')
plt.xlabel('Monto (USD millones)')
plt.legend()
plt.tight_layout()
plt.show()
```

Primero calcula media y mediana para usarlas en el gráfico. `bins=25` divide los datos en 25 barras. `kde=True` dibuja la curva de densidad suavizada encima. `plt.axvline()` dibuja una línea vertical — `color`, `linestyle='--'` (punteada) y `label` para la leyenda. `plt.legend()` muestra la leyenda con los labels definidos en cada `axvline`.

---

## CELDA 64 — Histograma de empleados

```python
media_emp   = df_clean['empleados'].mean()
mediana_emp = df_clean['empleados'].median()

plt.figure(figsize=(11, 5))
sns.histplot(data=df_clean, x='empleados', bins=30, kde=True)
plt.axvline(media_emp,   color='red',   linestyle='--', label=f'Media:   {media_emp:,.0f}')
plt.axvline(mediana_emp, color='black', linestyle='--', label=f'Mediana: {mediana_emp:,.0f}')
plt.title('Distribucion de empleados')
plt.xlabel('Empleados')
plt.legend()
plt.tight_layout()
plt.show()
```

Mismo patrón que la celda anterior pero para la variable `empleados`. `bins=30` usa 30 barras porque la variable tiene un rango mayor.

---

## CELDA 67 — Boxplot por etapa de ronda

```python
orden_rondas = ['Seed', 'Serie A', 'Serie B', 'Serie C', 'Serie D', 'Serie E', 'Serie F', 'Serie J']

plt.figure(figsize=(12, 5))
sns.boxplot(data=df_clean, x='ultima_ronda', y='valuacion_usd_mm',
            order=orden_rondas, palette='Blues')
plt.title('Valuacion por etapa de ronda')
plt.xlabel('Etapa')
plt.ylabel('Valuacion (USD millones)')
plt.tight_layout()
plt.show()
```

`orden_rondas` define el orden lógico de las etapas — sin esto seaborn las pondría en orden alfabético. `x='ultima_ronda'` pone las etapas en el eje horizontal, `y='valuacion_usd_mm'` la valuación en el vertical. `palette='Blues'` usa una paleta de azules. El boxplot muestra: línea central = mediana, caja = rango intercuartil (25%-75%), bigotes = rango normal, puntos = outliers.

---

## CELDA 68 — Boxplot por sector

```python
top_sectores = df_clean['sector'].value_counts().head(8).index

plt.figure(figsize=(11, 6))
sns.boxplot(data=df_clean[df_clean['sector'].isin(top_sectores)],
            x='empleados', y='sector', palette='Blues')
plt.title('Distribucion de empleados por sector')
plt.xlabel('Empleados')
plt.ylabel('')
plt.tight_layout()
plt.show()
```

Filtra solo los 8 sectores con más startups para que el gráfico no sea demasiado grande. Aquí `x` e `y` están invertidos respecto a la celda anterior — esto lo hace horizontal, más fácil de leer con los nombres de sector en el eje Y.

---

## CELDA 71 — Scatter plot

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='monto_ronda_usd_mm', y='valuacion_usd_mm',
                hue='ultima_ronda', palette='tab10', s=80, alpha=0.8)
plt.title('Monto de ronda vs Valuacion')
plt.xlabel('Monto ultima ronda (USD millones)')
plt.ylabel('Valuacion (USD millones)')
plt.legend(title='Ronda')
plt.tight_layout()
plt.show()
```

`x` e `y` son las dos variables numéricas a comparar. `hue='ultima_ronda'` colorea cada punto según la etapa de ronda. `s=80` es el tamaño de los puntos. `alpha=0.8` es la transparencia (0 = invisible, 1 = sólido) — útil cuando hay puntos superpuestos.

---

## CELDA 73 — Tabla de las startups más valiosas

```python
df_clean[['startup', 'ultima_ronda', 'monto_ronda_usd_mm', 'valuacion_usd_mm']].dropna().sort_values('valuacion_usd_mm', ascending=False).head(15)
```

`.dropna()` elimina filas con cualquier nulo en las columnas seleccionadas. Permite identificar exactamente qué startup es cada punto en el scatter — útil para el análisis en clase.

---

## CELDA 74 — Scatter sin OpenAI

```python
sin_openai = df_clean[df_clean['startup'] != 'OpenAI']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=sin_openai, x='monto_ronda_usd_mm', y='valuacion_usd_mm',
                hue='ultima_ronda', palette='tab10', s=80, alpha=0.8)
plt.title('Monto vs Valuacion — sin OpenAI')
plt.xlabel('Monto ultima ronda (USD millones)')
plt.ylabel('Valuacion (USD millones)')
plt.legend(title='Ronda')
plt.tight_layout()
plt.show()
```

`!=` significa "distinto de". Filtra todas las filas excepto OpenAI. Al sacarlo, la escala del gráfico baja de 300.000M a 65.000M y se puede ver la estructura del resto del mercado. Demuestra el impacto que tiene un solo outlier en la visualización.

---

## CELDA 76 — Scatter con regresión (lmplot)

```python
sns.lmplot(data=sin_openai, x='monto_ronda_usd_mm', y='valuacion_usd_mm',
           height=6, aspect=1.5,
           scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
plt.title('Monto vs Valuacion — con regresion')
plt.xlabel('Monto ultima ronda (USD millones)')
plt.ylabel('Valuacion (USD millones)')
plt.tight_layout()
plt.show()
```

`lmplot` es como `scatterplot` pero agrega automáticamente la línea de regresión lineal. `height` y `aspect` controlan el tamaño (es figure-level, no usa `plt.figure()`). `scatter_kws` y `line_kws` son diccionarios que pasan parámetros a los puntos y a la línea respectivamente. La línea muestra la tendencia general — si los puntos están muy dispersos alrededor de ella, la correlación es débil.

---

## CELDA 79 — Heatmap de correlación

```python
cols = ['anio_fundacion', 'monto_ronda_usd_mm', 'valuacion_usd_mm', 'empleados']
corr = df_clean[cols].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1)
plt.title('Matriz de Correlacion')
plt.tight_layout()
plt.show()
```

`.corr()` calcula la correlación entre todas las combinaciones de columnas numéricas — devuelve un número entre -1 y 1. `annot=True` muestra el número dentro de cada celda. `fmt='.2f'` formatea a 2 decimales. `cmap='RdYlGn'` define la paleta: rojo para negativo, amarillo para neutro, verde para positivo. `center=0` hace que el color neutro sea exactamente 0. `vmin=-1, vmax=1` usa la escala completa.

---

## CELDA 81 — Heatmap con todas las columnas numéricas

```python
corr = df_clean.select_dtypes(include='number').corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1)
plt.title('Matriz de Correlacion')
plt.tight_layout()
plt.show()
```

`select_dtypes(include='number')` selecciona automáticamente todas las columnas numéricas del DataFrame — no hay que listarlas a mano. Si se agrega una columna numérica nueva (como `edad_anios` o `es_unicornio`), aparece automáticamente en este heatmap.
