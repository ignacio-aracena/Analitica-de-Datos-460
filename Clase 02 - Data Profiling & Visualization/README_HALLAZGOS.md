# Informe de Hallazgos — Superstore 2014-2017

**Dataset:** Superstore.csv · 9.994 órdenes · Enero 2014 — Diciembre 2017
**Análisis realizado en:** `practica02.ipynb`

---

## El dataset

Superstore es una empresa retail de EEUU que vende productos en tres categorías (Technology, Furniture, Office Supplies) a través de tres segmentos de clientes (Consumer, Corporate, Home Office) en cuatro regiones geográficas (West, East, Central, South).

Cada fila del dataset es una orden individual. Las variables clave son:

- `Sales` — ingresos por la orden (en USD)
- `Profit` — ganancia neta (puede ser negativa si hubo pérdida)
- `Discount` — descuento aplicado (0 = sin descuento, 0.8 = 80% de descuento)
- `Quantity` — unidades vendidas

---

## 1. Calidad de los datos

**Hallazgo:** el 8% de las filas (799 de 9.994) tienen datos faltantes simultáneamente en `Postal Code`, `Discount` y `Profit`. El mismo conjunto de filas está incompleto en las tres columnas — lo que sugiere que son filas mal cargadas, no ausencias con patrón de negocio.

**Clasificación:** MCAR (Missing Completely at Random) — la ausencia no parece depender del valor de ninguna otra variable. No hay evidencia de que sean, por ejemplo, las órdenes más grandes o las de una región en particular.

**Decisión de imputación:**
- `Discount` → imputado con `0` (sin registro de descuento = sin descuento aplicado)
- `Profit` → imputado con la mediana (`$8.67`) — no se puede asumir 0 porque puede ser positivo o negativo
- `Postal Code` → dejado como nulo — no afecta el análisis

**Después de la limpieza:** el dataset tiene 9.994 filas utilizables en las variables analíticas principales.

---

## 2. Distribución de ventas

**Hallazgo central:** la distribución de ventas es fuertemente asimétrica hacia la derecha.

| Medida | Valor |
|---|---|
| Media | $229,86 |
| Mediana | $54,49 |
| Moda | $12,96 |
| Desviación estándar | $623,25 |

La media es **4,2 veces** la mediana. Esto significa que la mayoría de las órdenes son pequeñas (la mitad vale menos de $54), pero hay un grupo reducido de órdenes muy grandes que eleva el promedio artificialmente.

**Implicación:** reportar el ticket promedio ($230) da una imagen distorsionada del negocio. El valor más representativo de una orden típica es la mediana ($54).

La desviación estándar ($623) es 2,7 veces la media — un nivel de dispersión altísimo que confirma que el promedio no describe bien la realidad de las transacciones.

---

## 3. Análisis por categoría de producto

Esta es la pregunta central del análisis: **¿qué categoría genera más valor?**

| Categoría | Ventas totales | Profit total | Margen aproximado |
|---|---|---|---|
| Technology | ~$836K | ~$145K | ~17% |
| Furniture | ~$742K | ~$18K | ~2% |
| Office Supplies | ~$719K | ~$122K | ~17% |

**Hallazgos:**

**Technology** lidera tanto en ventas como en profit, y tiene el mejor margen del negocio. Es la categoría más rentable sin discusión.

**Office Supplies** vende menos que Furniture pero genera casi 7 veces más ganancia — su margen es comparable al de Technology.

**Furniture** es el hallazgo sorprendente: vende $742K (el segundo mayor volumen) pero genera apenas $18K de profit. Su margen es casi inexistente (~2%). El negocio está destinando recursos, inventario y logística a una categoría que apenas cubre costos.

> **La conclusión no era obvia antes del análisis.** Furniture parece un pilar del negocio por su volumen de ventas, pero en términos de rentabilidad es la categoría más débil por lejos.

---

## 4. El efecto de los descuentos sobre la rentabilidad

**Pregunta:** si Furniture tiene tan mal margen, ¿puede ser por la política de descuentos?

**Hallazgo:** existe una relación negativa clara entre el nivel de descuento aplicado y el profit generado. A mayor descuento, menor profit — y a partir de cierto umbral, el profit se vuelve negativo de forma sistemática.

**El umbral crítico está alrededor del 40% de descuento.** Las órdenes con descuentos superiores al 40-50% casi invariablemente generan pérdidas. Hay órdenes con descuentos del 70-80% que generan pérdidas de hasta $3.800 por transacción.

Esto no es ruido — es un patrón consistente en miles de órdenes.

**Implicación:** parte del bajo margen de Furniture probablemente está explicado por una política de descuentos agresiva. El negocio está subsidiando ventas a pérdida.

---

## 5. Análisis regional

| Región | Profit total | Margen % |
|---|---|---|
| West | Mayor | Mayor |
| East | Alto | Alto |
| South | Medio | Medio |
| Central | Menor | Menor |

**Hallazgo:** West lidera en profit absoluto y en margen. Central es el caso problemático: tiene ventas considerables pero el margen más bajo de las cuatro regiones.

Dos hipótesis para profundizar en un análisis futuro:
- ¿Central está vendiendo más productos de Furniture (baja rentabilidad)?
- ¿Los descuentos en Central son más altos que en el resto del país?

---

## 6. Análisis por segmento de cliente

| Segmento | Ventas totales | Profit total | Profit promedio por orden |
|---|---|---|---|
| Consumer | Mayor | Mayor (absoluto) | Medio |
| Corporate | Medio | Medio | Alto |
| Home Office | Menor | Menor (absoluto) | Similar a Corporate |

**Hallazgo:** Consumer es el segmento más grande en volumen, pero no necesariamente el más eficiente. Home Office, siendo el segmento más chico, tiene un profit promedio por orden comparable al Corporate.

**Implicación estratégica:** el crecimiento no tiene que venir necesariamente de conseguir más clientes Consumer. Profundizar en Corporate y Home Office podría generar más rentabilidad por cliente adquirido.

---

## 7. Outliers

La detección formal con la regla del IQR (Q1 − 1.5×IQR, Q3 + 1.5×IQR) encontró:

| Variable | Outliers | % del total | Límite superior |
|---|---|---|---|
| Sales | 1.167 | 11,7% | $498,93 |
| Profit | 1.943 | 19,4% | $61,02 |
| Discount | 769 | 7,7% | 0,50 (50%) |

El 19,4% de las órdenes son outliers en Profit — una proporción altísima que refleja la distribución muy dispersa de las ganancias. No son errores: son la realidad de un negocio donde el profit por orden varía enormemente dependiendo del producto y del descuento aplicado.

---

## Síntesis — los 3 hallazgos clave

**1. Technology es el motor del negocio.** Lidera en ventas y en rentabilidad. Si el negocio tiene que priorizar una categoría para crecer, es esta.

**2. Furniture genera volumen pero no ganancia.** Representa el 20% de las ventas pero apenas el 6% del profit. La pregunta estratégica es si vale la pena mantener esta categoría con la estructura de costos y descuentos actual.

**3. Los descuentos altos destruyen el profit de forma sistemática.** Hay un umbral claro alrededor del 40% a partir del cual el negocio entra en pérdida. Revisar la política de descuentos — especialmente en Furniture — es probablemente la palanca de rentabilidad más directa disponible.
