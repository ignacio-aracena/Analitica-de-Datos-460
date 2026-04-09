# Informe de Hallazgos — Campañas de Marketing Digital

**Dataset:** `campañas_marketing.csv` · 93 filas crudas → 89 filas limpias · Inversión publicitaria diaria por canal y ventas atribuidas
**Análisis realizado en:** `Clase04_DataCleaning_RegresionLineal.ipynb`

---

## Contexto y pregunta de negocio (CRISP-DM · Business Understanding)

Somos el equipo de datos de una marca **DTC (direct-to-consumer)** de e-commerce. El equipo de marketing nos pasó un export con la inversión publicitaria diaria por canal (Instagram, Google Ads, TikTok, Facebook) y las ventas atribuidas a cada día. Vienen a pedirnos ayuda porque están por cerrar el presupuesto del próximo trimestre y no tienen una manera clara de justificar cuánto invertir.

> **Pregunta de negocio:** ¿cuántos USD de venta generamos, en promedio, por cada USD invertido en marketing digital? ¿Podemos predecir las ventas a partir de la inversión?

Si logramos contestar esto con un modelo confiable, marketing puede pasar de "tirar plata y rezar" a **estimar el retorno esperado** antes de gastar.

### El dataset

Cada fila representa un día de una campaña en un canal y región. Las variables clave son:

- `inversion_usd` — inversión publicitaria del día (USD)
- `ventas_usd` — ventas atribuidas a esa inversión (USD) — **es el target del modelo**
- `canal` — canal publicitario (Instagram, Google Ads, TikTok, Facebook)
- `region` — región geográfica (CABA, GBA, Interior)
- `impresiones` — impresiones servidas
- `clicks` — clicks recibidos
- `fecha` — fecha de la campaña

### Cómo lo abordamos (CRISP-DM)

| Fase | Sección de este informe |
|---|---|
| **Business Understanding** | Pregunta de negocio (arriba) |
| **Data Understanding** | §1 — Calidad de los datos (diagnóstico) |
| **Data Preparation** | §1 — Decisiones de limpieza |
| **Modeling** | §3 — El modelo |
| **Evaluation** | §4 — Lectura de negocio · §5 — Limitaciones |
| *Deployment* | Fuera del scope de la clase |

---

## 1. Calidad de los datos (CRISP-DM · Data Understanding + Data Preparation)

El dataset llegó con **5 problemas** de calidad que tuvimos que resolver antes de modelar. Los listamos en el orden en que conviene atacarlos (no en el orden en que aparecen):

| Orden | Problema | Columna(s) afectada(s) | Magnitud |
|---|---|---|---|
| 1 | **Formato de moneda como texto** | `inversion_usd` | 88 valores con `$1,209` / `USD 980` / `$2,174.78`. **Lo arreglamos primero** porque mientras sea texto no podemos imputar nulos numéricos. |
| 2 | **Valores faltantes** | `inversion_usd`, `clicks` | 5 nulos en inversión, 4 en clicks |
| 3 | **Categorías inconsistentes** | `canal`, `region` | `canal` tenía 15 variantes para 4 valores reales; `region` tenía 8 variantes para 3 |
| 4 | **Outlier extremo** | `ventas_usd` | 1 valor de ~$45.000 (10× el promedio del resto) |
| 5 | **Filas duplicadas exactas** | toda la fila | 3 filas |

### Decisiones de limpieza

| Problema | Decisión | Por qué |
|---|---|---|
| Formato moneda | **Sacar `$`, `USD`, `,` y convertir a float** | El export del CRM trajo los montos como texto de presentación (`"$1,209"`, `"USD 980"`). Mientras siga siendo `object`, **no podemos calcular mediana, ni correlaciones, ni entrenar el modelo** — todas esas operaciones requieren tipo numérico. Y pandas trataría `"$1,209"` y `"$1209"` como categorías distintas. **Es la primera limpieza** porque las siguientes (imputación, IQR) la asumen ya hecha. |
| Nulos en numéricas | **Imputar con la mediana** (no media, no eliminar fila) | Tres preguntas detrás de esta decisión: **(a) ¿imputar o eliminar?** Tenemos 93 filas; eliminar 9 es tirar ~10% del dataset por un puñado de faltantes. En un dataset chico cada fila cuenta. **(b) ¿mediana o media?** La mediana es **robusta a outliers**: usa el orden de los datos, no la magnitud, así que un valor de $45.000 no la mueve. La media sí se va para arriba con un solo extremo. Como ya sabíamos que había un outlier en `ventas_usd`, la media hubiera contaminado las imputaciones. **(c) ¿por qué no 0?** Un faltante no significa "ese día no hubo inversión" — significa "no sabemos cuánto hubo". Imputar con 0 metería un sesgo sistemático. |
| Categorías | **Mapeo a nombres canónicos** con diccionario | `Instagram`, `instagram`, `IG` y `ig ` son **el mismo canal de negocio** para nosotros, pero para pandas son 4 valores distintos. Si no normalizamos, cualquier `groupby('canal')` reporta métricas fragmentadas (Instagram aparece 4 veces, cada una con un pedacito de las ventas) y los gráficos quedan ilegibles. Usamos **diccionario explícito** (no `.replace` con regex) porque es auditable: si mañana aparece un canal nuevo no mapeado, salta como `NaN` y lo detectás de inmediato. |
| Outlier de $45.000 | **Eliminar** (regla del IQR) | **Por qué IQR y no z-score**: el z-score usa media + desvío estándar, que están justamente influenciados por los outliers que queremos detectar (paradoja: el método se "ciega" en datasets contaminados). Además, el z-score asume distribución normal — y las ventas casi nunca son normales (tienen cola a la derecha). El IQR usa cuartiles, que son robustos por construcción y no asumen ninguna forma de distribución. Bonus: la regla `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` coincide con los bigotes del boxplot, así que el diagnóstico visual y el filtro programático dicen lo mismo. **Decisión de eliminar (no conservar)**: el segundo valor más alto del dataset es ~$8.500; un salto a $45.000 no es una transición suave, es un error de carga. En otros contextos un outlier puede ser una señal real (campaña viral, Black Friday) y conviene conservarlo — la **técnica detecta candidatos, la decisión final es de negocio**. |
| Duplicados | **Eliminar conservando la primera ocurrencia** | Filas idénticas suelen aparecer cuando un export se corre dos veces o cuando hay un join mal hecho upstream. Sumarlas inflaría artificialmente las ventas totales y le daría peso doble a esos días al entrenar el modelo. Conservamos la primera por convención (`keep="first"` de pandas) — daría lo mismo conservar la última porque son idénticas. |

**Después de la limpieza**: 89 filas utilizables, 0 nulos, 4 canales canónicos, 3 regiones canónicas, sin outliers extremos, sin duplicados.

> **Regla de oro**: sin datos limpios no hay análisis confiable. *Garbage in → garbage out.*

---

## 2. EDA — relaciones entre variables

Antes de modelar, miramos las correlaciones entre variables numéricas para elegir el predictor más fuerte.

| Par de variables | Correlación con `ventas_usd` | Interpretación |
|---|---|---|
| `inversion_usd` ↔ `ventas_usd` | **~0.97** (muy fuerte positiva) | Las ventas casi se mueven 1 a 1 con la inversión. Es nuestro mejor predictor. |
| `clicks` ↔ `ventas_usd` | moderada | Hay relación, pero más débil que con la inversión. |
| `impresiones` ↔ `ventas_usd` | débil | Las impresiones por sí solas no predicen bien las ventas (importa más cuántas se convierten en clicks). |

**Decisión de modelado**: **regresión lineal simple** con `inversion_usd` como único predictor. Una sola variable explicativa es suficiente para una primera respuesta accionable, y la fuerza de la correlación lo justifica.

---

## 3. El modelo — regresión lineal simple

**Configuración:**
- **X** (predictor): `inversion_usd`
- **y** (target): `ventas_usd`
- **Split**: 80% train (71 filas) / 20% test (18 filas), `random_state=42`
- **Algoritmo**: `LinearRegression` de scikit-learn

**Resultados — las tres métricas:**

| Métrica | Valor | Unidad | Qué responde |
|---|---|---|---|
| **MSE** | 126.874 | USD² | Error cuadrático promedio. **No es interpretable directamente** — nadie habla en "dólares al cuadrado". Lo mostramos porque es **la métrica que el modelo minimiza por dentro** cuando entrena (la regresión lineal busca los `m` y `b` que hacen el MSE más chico posible). El RMSE sale literalmente de su raíz cuadrada. |
| **RMSE** | **$356** | USD (la misma que `y`) | "¿Por cuánto se equivoca mi modelo?" → Cuando el modelo predice las ventas de un día cualquiera, **en promedio se equivoca por unos $356**, para arriba o para abajo. **Esta es la métrica que se le lleva al CEO.** |
| **R²** | **0.952** | Sin unidad (0 a 1) | "¿Qué tan buena es la señal que encontré?" → El modelo captura el **95% de la variabilidad** de las ventas. El otro 5% son cosas que la inversión sola no explica (promociones, estacionalidad, día de la semana, etc.). |

### Cómo leer cada métrica

**MSE (Mean Squared Error) — el paso previo al RMSE.** Para cada día del set de test, calculamos el error (predicho − real), lo elevamos al cuadrado (para que los errores positivos y negativos no se cancelen, y para penalizar más fuerte los errores grandes) y promediamos. La regla mental: *MSE = el número que el modelo optimiza. RMSE = el mismo número, pero traducido a una unidad que entiende un humano.*

**RMSE (Root Mean Squared Error) — el error en plata.** Es la raíz cuadrada del MSE, así que vuelve a estar en USD. **Lectura literal:** "el modelo se equivoca, en promedio, en ±$356". Pero ojo: **RMSE solo no significa nada — siempre hay que compararlo contra la magnitud típica de la variable que se predice.** Las ventas medianas del dataset son ~$4.700 → un error de $356 es **~7% del promedio**. Eso es bueno para una primera estimación. Si las ventas medianas fueran $400, ese mismo RMSE sería desastroso.

**R² (coeficiente de determinación) — qué tan bien explica el modelo.** Imaginate que NO tenés modelo: lo único que podés hacer para predecir las ventas de mañana es decir "el promedio histórico". Ese es el **baseline tonto**. R² mide **cuánto mejor es tu modelo que ese baseline**, en una escala de 0 a 1:
- **R² = 0** → el modelo no es mejor que predecir siempre el promedio. Inútil.
- **R² = 1** → el modelo predice perfecto, error cero.
- **R² = 0.95** → el modelo captura el 95% de la variabilidad de las ventas.

R² **no tiene unidad**. Por eso no sirve para decirle al CEO "nos equivocamos por X" — para eso está el RMSE. R² responde otra pregunta: **¿qué tan buena es la señal que encontré?**

> **R² puede ser negativo**: si el modelo es *peor* que predecir el promedio, R² < 0. Suena raro pero pasa con modelos mal especificados.

### Por qué necesitamos las dos (RMSE + R²)

Responden preguntas distintas y se complementan:

- **RMSE sin R²:** sé cuánto me equivoco, pero no sé si eso es "lo mejor que se podía hacer" o si me estoy perdiendo señal.
- **R² sin RMSE:** sé que el modelo es bueno relativo al promedio, pero no sé **cuánta plata** representa el error.
- **Las dos juntas:** "El modelo explica el 95% de la variabilidad (R²) y cuando se equivoca, lo hace por unos $356 (RMSE)." **Eso** es una evaluación completa.

| Métrica | Para quién | Cuándo la usás |
|---|---|---|
| **RMSE** | CEO / cliente / negocio | Cuando hay que comunicar el error en dólares. |
| **R²** | Equipo técnico | Para comparar modelos entre sí o evaluar la calidad relativa de la señal. |

> **Frase para fijar:** *RMSE te dice cuánto te equivocás. R² te dice qué tan buena es la señal que encontraste. Necesitás las dos.*

### Aclaración honesta sobre el R² = 0.95

Un R² de 0.95 es **altísimo y rara vez aparece en la vida real**. Este dataset está armado para que la relación sea limpia y veamos el flujo end-to-end. En negocio digital real, un R² de 0.6 ya es decente y 0.7-0.8 es muy bueno. **Si alguna vez ves un R² de 0.99 con datos reales, desconfiá** — probablemente hay *data leakage* (la `y` se metió de alguna forma en la `X`).

> **¿Por qué RMSE y no MAE?** MAE (Mean Absolute Error) es el promedio simple de errores en valor absoluto — más fácil de explicar, también en dólares. RMSE penaliza más los errores grandes (por el cuadrado). En negocio normalmente preferís RMSE porque equivocarte por $2.000 una vez es peor que equivocarte por $200 diez veces.

**Coeficientes aprendidos:**

$$\text{ventas} = 3.10 \cdot \text{inversión} + 543$$

| Coeficiente | Valor | Lectura de negocio |
|---|---|---|
| **m** (pendiente) | **$3.10** | **Por cada USD adicional invertido en marketing digital, las ventas aumentan en promedio $3.10.** Este es el número clave que se le lleva al equipo de marketing. |
| **b** (intercepto) | **$543** | Ventas estimadas con inversión cero. **Hay que tomarlo con pinzas**: el modelo no fue entrenado con valores cercanos a 0, así que es una extrapolación. No significa literalmente "vendemos $543 sin invertir". |

---

## 4. Lectura de negocio

**El modelo dice tres cosas accionables para el equipo de marketing:**

**1. La inversión publicitaria explica casi toda la variabilidad de las ventas (R² = 0.95).** No hace falta sumar más variables para tener una predicción razonable. Esto es bueno y es malo:
- **Bueno**: el modelo es simple y muy interpretable. Una sola variable basta.
- **Malo**: si la inversión deja de funcionar (saturación, fatiga del público), el modelo no tiene de dónde agarrarse para explicar las ventas. **No nos protege contra cambios estructurales del mercado.**

**2. El ROI marginal es de ~3.1x.** Por cada peso adicional invertido, esperamos ~$3.10 de ventas. **Atención**: esto es una **estimación lineal promedio**. En la realidad, el ROI casi nunca es lineal hasta el infinito — hay un punto de saturación donde invertir más deja de dar retorno proporcional. La regresión lineal **no detecta ese punto** porque asume relación lineal.

**3. El modelo se equivoca en ±$356 al predecir un día.** En el contexto del dataset (ventas medianas de ~$4.700), eso es ~7% de error promedio. **Aceptable para una primera estimación, no para tomar decisiones de presupuesto sin un margen de seguridad.**

---

## 5. Limitaciones del modelo

Algunas cosas que **este modelo NO puede responder**:

- **¿Qué canal es más rentable?** El modelo no usa `canal` como variable. Para responder esto necesitamos una **regresión múltiple** que incluya canal como variable categórica (one-hot encoding).
- **¿Hay diferencias por región?** Mismo problema — `region` no está en el modelo.
- **¿Qué pasa si la inversión es muy alta?** Por la naturaleza lineal, el modelo extrapola sin freno. En la realidad probablemente hay un techo (saturación de la audiencia) que el modelo no captura.
- **¿La predicción es robusta en el tiempo?** Entrenamos con datos de un período corto. Si el comportamiento del consumidor cambia (estacionalidad, crisis, lanzamiento de un competidor), el modelo deja de ser válido. **Un modelo entrenado en febrero puede fallar en diciembre.**
- **¿Importan los clicks o las impresiones?** Las dejamos afuera del modelo aunque las tenemos en el dataset. Una regresión múltiple podría revelar si suman o si son redundantes con la inversión.

---

## 6. Síntesis — los 3 hallazgos clave

**1. La inversión publicitaria es el predictor dominante de las ventas (R² = 0.95).** Una regresión lineal simple con una sola variable basta para explicar el 95% de la variabilidad. **No siempre es así** — en otros datasets vas a necesitar varios predictores para llegar a este nivel de ajuste.

**2. El ROI marginal estimado es ~$3.10 por USD invertido.** Es una respuesta clara y comunicable al equipo de marketing. Pero hay que recordar dos asteriscos: **(a)** es un promedio, no aplica igual en todos los rangos de inversión; **(b)** el modelo no detecta saturación.

**3. La calidad de los datos no era un detalle menor.** El dataset crudo tenía 5 problemas serios — sin la limpieza de la Parte A, hubiéramos terminado con un modelo entrenado sobre `$1,209` (string) o sobre filas duplicadas inflando ventas, o sobre un outlier tirando la regresión. **El 80% del trabajo de un proyecto de datos es la Parte A.** Esa es la lección más importante de la clase.

---

## 7. Conclusión — volviendo a la pregunta de negocio

Empezamos preguntándonos dos cosas. Después del flujo completo (limpieza → EDA → modelo → evaluación), las podemos contestar de frente:

**1. ¿Cuántos USD de venta generamos, en promedio, por cada USD invertido en marketing digital?**

→ Aproximadamente **$3.10 por cada USD invertido**. Es el coeficiente `m` de la recta y representa el **ROI marginal estimado**. Es el número que se le lleva al equipo de marketing para justificar el presupuesto.

**2. ¿Podemos predecir las ventas a partir de la inversión?**

→ **Sí, con razonable confianza para una primera estimación.** El modelo:
- Explica el **95% de la variabilidad** de las ventas (R² = 0.95).
- Cuando se equivoca, lo hace por **±$356** en promedio (RMSE), que es ~**7%** de las ventas medianas del dataset.

Eso es suficiente para **estimar el retorno esperado antes de gastar** — exactamente lo que el equipo de marketing pidió. Marketing pasa de "tirar plata y rezar" a tener una expectativa cuantificada del retorno.

### Tres asteriscos antes de usar el modelo para cerrar el presupuesto del trimestre

1. **El ROI de $3.10 es un promedio lineal.** La regresión asume que cada dólar adicional rinde lo mismo, hasta el infinito. En la realidad hay un punto de saturación donde invertir más deja de dar retorno proporcional, y este modelo no lo detecta. Conclusión práctica: **el modelo es confiable dentro del rango de inversión observado en el dataset; extrapolar muy por encima es riesgoso.**

2. **El modelo no diferencia por canal ni por región.** Para responder "¿en qué canal conviene invertir el próximo dólar?" hace falta una regresión múltiple con `canal` y `region` como variables (one-hot encoding). Es el siguiente paso natural y queda como mejora pendiente (ver §5).

3. **El R² = 0.95 es excepcionalmente alto** porque el dataset está armado para enseñar el flujo limpio. En proyectos reales, esperar valores entre 0.6 y 0.8 es realista. **No usar este R² como vara de comparación para futuros modelos.**

### Recomendación al equipo de marketing

Usar el modelo como **guía direccional** (cuánto rinde aproximadamente cada peso invertido) y como **input para construir escenarios de presupuesto**, pero acompañarlo de un margen de seguridad (≥7%) y revalidarlo cada trimestre con datos nuevos. **No tratarlo como una bola de cristal** — es una primera respuesta cuantitativa, no la palabra final.

---

## Recursos relacionados

- **`Lineal_Simple_Regression.ipynb`** (en esta misma carpeta) — notebook de ejemplo con el mismo flujo de regresión lineal aplicado a otro dataset (salarios vs años de experiencia, autoría G. Vinueza). Incluye un paso adicional con `statsmodels` para extraer **p-values** y **F-statistic**, que sirven para validar si los coeficientes son estadísticamente significativos. Recomendado si querés ver cómo se profundiza en la validación estadística del modelo.
- **`documentacion_tecnica.md`** — explicación celda por celda del código del notebook.
- **`Clase 4 - Data Cleaning & Linear Regression.pdf`** — slides teóricas de la clase de Pablo.
