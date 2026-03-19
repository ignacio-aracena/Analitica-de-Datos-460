# README Técnico — practica02.ipynb

Documentación del código celda por celda: qué hace, qué produce, qué parámetros usa y por qué.

---

## Stack de dependencias

```python
pandas      # manipulación de datos (DataFrames, groupby, fillna, describe, pivot_table)
numpy       # operaciones numéricas (quantile, var, std usadas implícitamente por pandas)
matplotlib  # renderizado base de gráficos (pyplot, axes, fig)
seaborn     # gráficos estadísticos de alto nivel (histplot, boxplot, scatterplot)
```

Versiones recomendadas: `pandas >= 1.5`, `matplotlib >= 3.5`, `seaborn >= 0.12`

---

## 0. Setup

### Celda 1 — Imports y configuración global

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
pd.set_option('display.float_format', '{:.2f}'.format)
```

| Línea | Efecto |
|---|---|
| `sns.set_theme(style='whitegrid', palette='muted')` | Aplica tema global a todos los gráficos del notebook — fondo blanco con grilla gris suave, colores pastel por defecto |
| `pd.set_option('display.float_format', '{:.2f}'.format)` | Formatea todos los floats a 2 decimales en outputs de DataFrames — evita notación científica |

> `sns.set_theme` actúa globalmente: afecta todos los gráficos de matplotlib/seaborn creados después de esta línea, incluso los que no usan seaborn directamente.

---

### Celda 2 — Carga del dataset

```python
df = pd.read_csv('Superstore.csv', encoding='latin', parse_dates=['Order Date', 'Ship Date'])
print(f'Filas: {df.shape[0]} | Columnas: {df.shape[1]}')
df.head()
```

| Parámetro | Valor | Por qué |
|---|---|---|
| `encoding='latin'` | latin-1 / ISO-8859-1 | El CSV tiene caracteres especiales no-UTF8 — sin este parámetro, pandas lanza `UnicodeDecodeError` |
| `parse_dates=['Order Date', 'Ship Date']` | Lista de columnas | Convierte esas columnas de `object` a `datetime64[ns]` en la carga — evita hacer la conversión manual después |

**Output:** `Filas: 9994 | Columnas: 21` + preview de primeras 5 filas.

**DataFrame resultante `df`:**
- Shape: `(9994, 21)`
- Tipos: `datetime64[ns]` × 2, `float64` × 4, `int64` × 2, `object` × 13
- Nulos al cargar: 799 en `Postal Code`, `Discount` y `Profit` (mismas filas)

---

## Paso 2 — Exploración

### Celda 3 — `df.info()`

```python
df.info()
```

**Output:** tabla de columnas con dtype y Non-Null Count para cada una.

**Qué mirar:**
- Columnas con `Non-Null Count < 9994` → tienen nulos
- Columnas `object` que deberían ser numéricas o fechas → necesitan conversión de tipo

---

### Celda 4 — Cardinalidad de variables categóricas

```python
cat_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
for col in cat_cols:
    print(f'{col}: {df[col].nunique()} valores únicos → {df[col].unique()}')
```

**Métodos usados:**
- `nunique()` — cuenta valores distintos en la columna (excluye NaN)
- `unique()` — devuelve array numpy con los valores únicos

**Output:** por cada columna, cantidad de categorías y sus nombres.

| Columna | Categorías |
|---|---|
| Ship Mode | 4 |
| Segment | 3 |
| Region | 4 |
| Category | 3 |
| Sub-Category | 17 |

---

### Celda 5 — `df.describe()`

```python
df.describe().T
```

**`.T` (transpuesta):** intercambia filas y columnas — con 21 columnas, el output horizontal es ilegible; transpuesto, cada fila es una variable.

**Columnas del output:** `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`

**Nota:** `describe()` incluye columnas datetime cuando están en el DataFrame. Para fechas, muestra min/max/percentiles como timestamps; `mean` y `std` aparecen como NaN para fechas.

---

### Celda 6 — Detección de nulos

```python
nulos = df.isnull().sum()
nulos_pct = (df.isnull().mean() * 100).round(1)
resumen = pd.DataFrame({'Nulos': nulos, '% del total': nulos_pct})
resumen[resumen['Nulos'] > 0].sort_values('Nulos', ascending=False)
```

| Línea | Qué hace |
|---|---|
| `df.isnull()` | DataFrame booleano: `True` donde hay NaN |
| `.sum()` | Suma por columna — cuenta Trues (= cantidad de nulos) |
| `.mean() * 100` | Promedio por columna (0-1) × 100 = porcentaje de nulos |
| `.round(1)` | Redondea a 1 decimal |
| `pd.DataFrame({...})` | Crea un nuevo DataFrame con las dos series como columnas |
| `resumen[resumen['Nulos'] > 0]` | Filtra: solo columnas que tienen al menos 1 nulo |
| `.sort_values('Nulos', ascending=False)` | Ordena de más nulos a menos |

**Output:**

| Columna | Nulos | % del total |
|---|---|---|
| Postal Code | 799 | 8.0 |
| Discount | 799 | 8.0 |
| Profit | 799 | 8.0 |

---

### Celda 7 — Gráfico de barras de nulos

```python
plt.figure(figsize=(8, 4))
cols_con_nulos = nulos_pct[nulos_pct > 0].sort_values(ascending=False)
bars = plt.bar(cols_con_nulos.index, cols_con_nulos.values, color='salmon', edgecolor='white')
for bar, val in zip(bars, cols_con_nulos.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val}%', ha='center', fontsize=11)
plt.title('Porcentaje de valores faltantes por columna')
plt.ylabel('% Faltantes')
plt.ylim(0, 12)
plt.tight_layout()
plt.show()
```

**Técnica de anotación sobre barras:**
- `bar.get_x() + bar.get_width()/2` → coordenada X del centro de cada barra
- `bar.get_height() + 0.1` → coordenada Y justo encima del tope de la barra
- `ha='center'` → alineación horizontal centrada

**Parámetros visuales:** `figsize=(8,4)`, `color='salmon'`, `ylim=(0,12)` forzado para dar espacio a las etiquetas.

---

### Celda 8 — Imputación de nulos

```python
df['Discount'] = df['Discount'].fillna(0)
df['Profit']   = df['Profit'].fillna(df['Profit'].median())

print('Nulos restantes:')
print(df.isnull().sum()[df.isnull().sum() > 0])
```

| Columna | Estrategia | Valor imputado |
|---|---|---|
| `Discount` | `fillna(0)` | 0 — asume que sin registro = sin descuento |
| `Profit` | `fillna(df['Profit'].median())` | `8.67` — mediana calculada sobre los 9195 valores no nulos |
| `Postal Code` | No se imputa | Se deja como NaN — no afecta el análisis |

**Nota:** `df['Profit'].median()` se evalúa **antes** de `fillna` — calcula la mediana sobre los 9195 valores no nulos y usa ese escalar como valor de imputación.

**Output post-imputación:**
```
Nulos restantes:
Postal Code    799
```

---

## Paso 3 — Insight

### Celda 9 — Media, mediana y moda de ventas

```python
ventas = df['Sales'].dropna()

media   = ventas.mean()
mediana = ventas.median()
moda    = ventas.mode()[0]

print(f'Media de ventas:   ${media:,.2f}')
print(f'Mediana de ventas: ${mediana:,.2f}')
print(f'Moda de ventas:    ${moda:,.2f}')
print(f'Diferencia media-mediana: ${media - mediana:,.2f}')
```

**Métodos:**
- `.mean()` → promedio aritmético
- `.median()` → valor central (percentil 50)
- `.mode()` → devuelve una **Serie** con todos los valores modales; `[0]` toma el primero

**Output:**
```
Media de ventas:   $229.86
Mediana de ventas: $54.49
Moda de ventas:    $12.96
Diferencia media-mediana: $175.37
```

**Por qué `mode()` devuelve una Serie:** si hay empate (dos valores con la misma frecuencia), pandas devuelve todos. `[0]` garantiza que siempre se obtiene un escalar.

---

### Celda 10 — Histograma con media/mediana/moda

```python
plt.figure(figsize=(10, 5))
sns.histplot(ventas, bins=50, kde=True, color='steelblue')
plt.axvline(media,   color='red',    linestyle='--', linewidth=2, label=f'Media   ${media:,.0f}')
plt.axvline(mediana, color='green',  linestyle='--', linewidth=2, label=f'Mediana ${mediana:,.0f}')
plt.axvline(moda,    color='orange', linestyle='--', linewidth=2, label=f'Moda    ${moda:,.0f}')
plt.title('Distribución de Ventas — Media, Mediana y Moda')
plt.xlabel('Ventas (USD)')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.show()
```

| Parámetro | Valor | Efecto |
|---|---|---|
| `bins=50` | 50 intervalos | Granularidad suficiente para ver la forma de la distribución sin ruido excesivo |
| `kde=True` | Kernel Density Estimate | Superpone curva de densidad suavizada sobre el histograma |
| `plt.axvline` | Línea vertical en x | Marca la posición de media/mediana/moda en el eje X |
| `linestyle='--'` | Línea punteada | Permite distinguirlas sin tapar las barras |

**Output:** histograma con sesgo visible a la derecha — la media queda muy a la derecha de la mediana por efecto de órdenes grandes.

---

### Celda 11 — Ventas y profit por categoría (`groupby`)

```python
resumen_cat = df.groupby('Category')[['Sales', 'Profit']].sum().sort_values('Sales', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

resumen_cat['Sales'].plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Ventas totales por categoría')
axes[0].set_xlabel('Ventas (USD)')
axes[0].set_ylabel('')

resumen_cat['Profit'].plot(kind='barh', ax=axes[1], color='seagreen')
axes[1].set_title('Profit total por categoría')
axes[1].set_xlabel('Profit (USD)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

**Cadena `groupby`:**
1. `df.groupby('Category')` → agrupa las 9994 filas en 3 grupos (Furniture, Office Supplies, Technology)
2. `[['Sales', 'Profit']]` → selecciona solo esas dos columnas
3. `.sum()` → suma dentro de cada grupo → DataFrame de shape `(3, 2)`
4. `.sort_values('Sales', ascending=False)` → ordena por ventas descendente

**Resultado `resumen_cat`:**

| Category | Sales | Profit |
|---|---|---|
| Technology | ~836K | ~145K |
| Furniture | ~742K | ~18K |
| Office Supplies | ~719K | ~122K |

**Gráfico:** `plt.subplots(1, 2)` crea una figura con 2 subplots en fila. `kind='barh'` hace barras horizontales — más legibles cuando las etiquetas de categorías son largas.

---

### Celda 12 — Scatter Discount vs Profit

```python
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x='Discount', y='Profit', alpha=0.4, color='steelblue')
plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Profit = 0')
plt.title('Relación entre Descuento y Profit')
plt.xlabel('Descuento aplicado')
plt.ylabel('Profit (USD)')
plt.legend()
plt.tight_layout()
plt.show()
```

| Parámetro | Valor | Efecto |
|---|---|---|
| `alpha=0.4` | Transparencia 40% | Con 9994 puntos, sin transparencia los puntos se solapan completamente |
| `plt.axhline(0, ...)` | Línea horizontal en y=0 | Marca el umbral de pérdida — puntos debajo = órdenes con profit negativo |

**Insight visible:** los puntos se concentran bajo el eje 0 a medida que `Discount` supera 0.4 → relación negativa entre descuento alto y profit.

---

## Paso 4 — Visualización

### Celda 13 — Bad vs Good chart (side-by-side)

```python
cat_sales = df.groupby('Category')['Sales'].sum()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── GRAFICO MAL ──────────────────────────────────────────────────────────
axes[0].bar(cat_sales.index, cat_sales.values,
            color=['#e74c3c', '#f39c12', '#2ecc71'])
axes[0].set_title('Sales', fontsize=12)
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Quarterly Sales In Thousands ($) 2014-2017 Fiscal Year')
axes[0].tick_params(axis='x', rotation=30)

# ── GRAFICO BIEN ─────────────────────────────────────────────────────────
cat_sorted = cat_sales.sort_values(ascending=True)
bars = axes[1].barh(cat_sorted.index, cat_sorted.values, color='steelblue')
for bar, val in zip(bars, cat_sorted.values):
    axes[1].text(bar.get_width() + 8000, bar.get_y() + bar.get_height() / 2,
                 f'${val/1e6:.2f}M', va='center', fontsize=10)
axes[1].set_xlabel('Ventas totales (USD)')
axes[1].set_xlim(0, cat_sorted.max() * 1.25)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
```

**Errores deliberados en el gráfico de la izquierda:**
- `color=['#e74c3c', '#f39c12', '#2ecc71']` — 3 colores distintos sin significado semántico
- `set_title('Sales')` — título que no comunica nada
- `set_ylabel('Quarterly Sales...')` — eje con 60 caracteres
- Orden original (alfabético), barras verticales con etiquetas rotadas

**Buenas prácticas en el gráfico de la derecha:**
- `sort_values(ascending=True)` — se ordena ascendente para que la barra más larga quede arriba en el barh
- `barh` — horizontal, más legible
- `axes[1].text(bar.get_width() + 8000, ...)` — valor anotado en millones fuera del borde de cada barra
- `val/1e6:.2f}M` — formatea a millones con 2 decimales + sufijo M
- `set_xlim(0, cat_sorted.max() * 1.25)` — extiende el eje 25% para que quepan las anotaciones
- `spines['top'].set_visible(False)` / `spines['right'].set_visible(False)` — elimina bordes superiores y derecho (reduce ruido visual)

---

## Extra 1 — Dispersión en profundidad

### Celda 14 — Varianza y desviación estándar

```python
varianza = ventas.var()
std      = ventas.std()

print(f'Varianza:            {varianza:>12,.0f}  (difícil de interpretar — unidades al cuadrado)')
print(f'Desviación estándar: ${std:>11,.2f}  (en USD, comparables con Sales)')
print(f'Media:               ${media:>11,.2f}')
print()
print(f'La std (${std:,.0f}) es {std/media:.1f}x la media → distribución muy dispersa.')
```

**Métodos:**
- `.var()` → varianza muestral (divide por `n-1`, no `n`) — es el default en pandas
- `.std()` → desviación estándar muestral (raíz de la varianza muestral)

**Output:**
```
Varianza:               388,434  (difícil de interpretar — unidades al cuadrado)
Desviación estándar: $     623.25  (en USD, comparables con Sales)
Media:               $     229.86

La std ($623) es 2.7x la media → distribución muy dispersa.
```

---

### Celda 15 — Boxplots de tres variables

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, col, color in zip(axes, ['Sales', 'Profit', 'Discount'], ['steelblue', 'seagreen', 'salmon']):
    data = df[col].dropna()
    sns.boxplot(y=data, ax=ax, color=color, width=0.4,
                flierprops=dict(marker='o', color='red', alpha=0.4, markersize=4))
    ax.set_title(col)
    ax.set_ylabel('')

plt.suptitle('Distribución de variables numéricas clave — outliers en rojo', fontsize=13)
plt.tight_layout()
plt.show()
```

**`zip(axes, [...], [...])`:** itera los 3 subplots simultáneamente con su columna y color — evita duplicar el código 3 veces.

**Parámetros del boxplot:**

| Parámetro | Valor | Efecto |
|---|---|---|
| `y=data` | Serie | Boxplot vertical (si fuera `x`, sería horizontal) |
| `width=0.4` | float | Ancho de la caja — más angosto que el default (0.8) |
| `flierprops=dict(marker='o', color='red', alpha=0.4, markersize=4)` | dict | Estilo de los puntos outliers — círculos rojos semitransparentes, pequeños |

**Qué muestra cada boxplot:**
- Caja: rango intercuartil (Q1-Q3, el 50% central)
- Línea interior: mediana (Q2)
- Bigotes: hasta Q1−1.5×IQR y Q3+1.5×IQR
- Puntos rojos: outliers (fuera de los bigotes)

---

### Celda 16 — Función `resumen_outliers`

```python
def resumen_outliers(serie, nombre):
    s = serie.dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    outliers = s[(s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)]
    print(f'{nombre}: {len(outliers)} outliers ({100*len(outliers)/len(s):.1f}%) '
          f'| Límites: [{Q1 - 1.5*IQR:.2f}, {Q3 + 1.5*IQR:.2f}]')

resumen_outliers(df['Sales'],    'Sales   ')
resumen_outliers(df['Profit'],   'Profit  ')
resumen_outliers(df['Discount'], 'Discount')
```

**Lógica interna:**

| Línea | Operación |
|---|---|
| `s.quantile(0.25)`, `s.quantile(0.75)` | Q1 y Q3 — percentiles 25 y 75 |
| `IQR = Q3 - Q1` | Rango intercuartil |
| `s < Q1 - 1.5*IQR` | Máscara booleana: outliers inferiores |
| `s > Q3 + 1.5*IQR` | Máscara booleana: outliers superiores |
| `|` | OR lógico entre las dos máscaras |
| `s[...]` | Filtrado — devuelve solo los outliers |

**Output:**
```
Sales   : 1167 outliers (11.7%) | Límites: [-271.71, 498.93]
Profit  : 1943 outliers (19.4%) | Límites: [-32.93, 61.02]
Discount: 769 outliers  (7.7%)  | Límites: [-0.30, 0.50]
```

---

## Extra 2 — Más preguntas de negocio

### Celda 17 — Performance por región

```python
region_perf = df.groupby('Region')[['Sales', 'Profit']].sum()
region_perf['Profit Margin %'] = (region_perf['Profit'] / region_perf['Sales'] * 100).round(1)
region_perf = region_perf.sort_values('Profit', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ['seagreen' if x > 0 else 'salmon' for x in region_perf['Profit']]
region_perf['Profit'].plot(kind='bar', ax=axes[0], color=colors, edgecolor='white')
axes[0].set_title('Profit total por región')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=0)

region_perf['Profit Margin %'].plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='white')
axes[1].set_title('Margen de profit por región (%)')
axes[1].set_ylabel('Margen %')
axes[1].tick_params(axis='x', rotation=0)
```

**Columna calculada:**
```python
region_perf['Profit Margin %'] = (region_perf['Profit'] / region_perf['Sales'] * 100).round(1)
```
Margen % = Profit total / Sales total — operación vectorizada sobre Series del DataFrame agrupado.

**Color condicional:**
```python
colors = ['seagreen' if x > 0 else 'salmon' for x in region_perf['Profit']]
```
List comprehension que asigna color según si el profit es positivo o negativo — aquí todas las regiones son positivas, pero el patrón es reutilizable.

---

### Celda 18 — Ventas y profit por segmento

```python
seg = df.groupby('Segment')[['Sales', 'Profit']].agg(['sum', 'mean']).round(2)
seg.columns = ['Ventas totales', 'Venta promedio', 'Profit total', 'Profit promedio']
seg = seg.sort_values('Profit total', ascending=False)

plt.figure(figsize=(9, 5))
x = range(len(seg))
width = 0.35
plt.bar([i - width/2 for i in x], seg['Ventas totales'], width, label='Ventas totales', color='steelblue')
plt.bar([i + width/2 for i in x], seg['Profit total'],  width, label='Profit total',  color='seagreen')
plt.xticks(x, seg.index)
```

**`agg(['sum', 'mean'])`:** aplica múltiples funciones de agregación en una sola llamada. Produce un MultiIndex en las columnas (`Sales`/`Profit` × `sum`/`mean`).

**Aplanado del MultiIndex:**
```python
seg.columns = ['Ventas totales', 'Venta promedio', 'Profit total', 'Profit promedio']
```
Reemplaza el MultiIndex con nombres planos — más cómodo para acceder luego.

**Barras agrupadas manuales:**
- `[i - width/2 for i in x]` → posición del primer grupo (izquierda del centro)
- `[i + width/2 for i in x]` → posición del segundo grupo (derecha del centro)
- `plt.xticks(x, seg.index)` → pone las etiquetas de categoría en el centro de cada par

---

## Extra 3 — Barras apiladas absolutas vs porcentaje

### Celda 19 — `pivot_table` + barras apiladas

```python
pivot = df.pivot_table(values='Sales', index='Region', columns='Category', aggfunc='sum')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

pivot.plot(kind='bar', ax=axes[0], stacked=True)
axes[0].set_title('Ventas absolutas por región y categoría')
axes[0].tick_params(axis='x', rotation=0)

pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pivot_pct.plot(kind='bar', ax=axes[1], stacked=True)
axes[1].set_title('Composición % por región y categoría')
axes[1].set_ylabel('%')
axes[1].tick_params(axis='x', rotation=0)
```

**`pivot_table`:**

| Parámetro | Valor | Efecto |
|---|---|---|
| `values='Sales'` | Columna a agregar | Los valores en las celdas son ventas |
| `index='Region'` | Filas de la tabla | Una fila por región |
| `columns='Category'` | Columnas de la tabla | Una columna por categoría |
| `aggfunc='sum'` | Función | Suma las ventas donde coinciden región + categoría |

**Resultado `pivot`:** DataFrame de shape `(4, 3)` — 4 regiones × 3 categorías.

**Normalización a porcentaje:**
```python
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
```
- `pivot.sum(axis=1)` → suma por fila (total ventas por región) → Serie de longitud 4
- `.div(..., axis=0)` → divide cada fila por su total → valores 0-1
- `* 100` → convierte a porcentaje 0-100

**`stacked=True`:** apila las categorías sobre el mismo eje — cada barra llega hasta el total.

---

## Resumen de outputs del notebook

| Celda | Output |
|---|---|
| Setup | — |
| Carga | `(9994, 21)` + df.head() |
| `info()` | Tabla de tipos y non-null counts |
| Cardinalidad | 5 líneas de texto |
| `describe()` | Tabla transpuesta 8×9 |
| Nulos tabla | DataFrame 3×2 |
| Nulos gráfico | Barras salmon |
| Imputación | Confirmación de nulos restantes |
| Media/mediana/moda | 4 líneas de texto |
| Histograma | Distribución sesgada con 3 líneas verticales |
| Groupby categoría | 2 barras horizontales |
| Scatter | Nube de puntos con línea en y=0 |
| Bad vs good | Side-by-side — mismo dato, dos visualizaciones |
| Varianza/std | 4 líneas de texto |
| Boxplots | 3 boxplots — Sales, Profit, Discount |
| `resumen_outliers()` | 3 líneas: outlier count + límites IQR |
| Región | 2 barras verticales + print del DataFrame |
| Segmento | Barras agrupadas + print del DataFrame |
| Barras apiladas | Side-by-side absolutas vs % |

---

## Convenciones de código usadas en el notebook

| Convención | Ejemplo | Por qué |
|---|---|---|
| F-strings con `:,.2f` | `f'${media:,.2f}'` | Formatea floats con separador de miles (,) y 2 decimales |
| `figsize` siempre explícito | `plt.figure(figsize=(10, 5))` | Control sobre el tamaño de salida en el notebook |
| `plt.tight_layout()` siempre al final | `plt.tight_layout()` | Evita que los títulos y ejes se superpongan |
| `ax=axes[i]` para subplots | `resumen_cat['Sales'].plot(kind='barh', ax=axes[0])` | Dirige el gráfico al subplot correcto |
| `dropna()` antes de calcular | `ventas = df['Sales'].dropna()` | Evita que los NaN afecten a mean/median/std |
| Variables intermedias nombradas | `resumen_cat`, `region_perf`, `pivot_pct` | Facilita debugging y relectura |
