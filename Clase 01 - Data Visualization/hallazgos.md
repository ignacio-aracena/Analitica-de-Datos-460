# Informe de Hallazgos — Startups de IA 2024-2025

**Dataset:** startups.csv · Empresas de IA (Crunchbase / TechCrunch / PitchBook, 2024-2025)
**Análisis realizado en:** `practica01.ipynb`

---

## El dataset

Startups de inteligencia artificial con datos reales de rondas de inversión recientes. Incluye empresas como OpenAI, Anthropic, xAI, DeepSeek, Mistral, Cursor, Canva, Waymo y Figure AI, entre otras.

Cada fila es una empresa. Las variables clave son:

- `monto_ronda_usd_mm` — monto levantado en la última ronda (USD millones)
- `valuacion_usd_mm` — valuación al momento de esa ronda (USD millones)
- `empleados` — cantidad de empleados
- `sector` — categoría (LLM, Robótica, Coding AI, etc.)
- `ultima_ronda` — etapa de inversión (Seed, Serie A/B/C...)
- `pais` — país de origen

---

## 1. Calidad de los datos

**Nulos encontrados:**

| Columna | % nulos | Razón |
|---|---|---|
| `valuacion_usd_mm` | ~27% | DeepSeek es autofinanciada; algunas startups no divulgan valuación pública |
| `monto_ronda_usd_mm` | ~10% | Midjourney nunca levantó ronda externa; empresas públicas no reportan monto reciente |

Estos nulos son **datos faltantes reales**, no errores de carga. La ausencia en sí misma es información — significa que la empresa opera fuera del modelo tradicional de venture capital.

**Duplicados:** se encontraron filas duplicadas y se eliminaron conservando la primera ocurrencia. El dataset limpio (`df_clean`) es el que se usa en todo el análisis posterior.

---

## 2. Distribución de valuaciones y montos de ronda

**Hallazgo central:** todas las variables numéricas tienen distribución fuertemente sesgada a la derecha.

| Variable | Media | Mediana | Ratio media/mediana |
|---|---|---|---|
| `monto_ronda_usd_mm` | ~USD 740M | ~USD 231M | 3,2× |
| `valuacion_usd_mm` | ~USD 14.700M | ~USD 3.000M | 4,9× |
| `empleados` | ~654 | ~250 | 2,6× |

**Los culpables del sesgo:**

- **OpenAI** — valuación de USD 300.000M. Un solo punto que desplaza la media de todo el dataset hacia arriba
- **Databricks** — ronda de USD 10.000M (la más grande del dataset), con valuación de USD 62.000M
- **xAI** — USD 6.000M de ronda y USD 50.000M de valuación en Serie C — alto para esa etapa

**Implicación:** la media de valuación (~USD 14.700M) no describe a ninguna empresa real del dataset. La mediana (~USD 3.000M) es el número honesto para hablar del "startup de IA típico".

---

## 3. Distribución geográfica

USA domina de forma aplastante el ecosistema de IA:

| País | Participación |
|---|---|
| USA | ~71% |
| Israel | ~6% |
| UK | ~5% |
| China | ~5% |
| Resto del mundo | ~13% |

El ecosistema de IA es altamente concentrado — más de 7 de cada 10 startups de este dataset son estadounidenses. Israel y UK son los únicos países con participación relevante fuera de USA y China.

---

## 4. Distribución por sector

Los sectores con más empresas en el dataset son los de mayor inversión reciente: LLMs, Coding AI y Robótica lideran en cantidad. Los sectores más chicos (como AI de salud o legal tech) tienen pocas empresas pero valuaciones medianas altas — están atrayendo capital concentrado en pocos jugadores.

---

## 5. Valuación por etapa de ronda

**Hallazgo:** la valuación mediana crece consistentemente con la etapa de ronda, desde Seed hasta Serie J. El patrón es claro: cada ronda valida y amplía la valuación anterior.

**Excepción notable:** xAI está en Serie C con una valuación de USD 50.000M — un múltiplo muy superior al típico de esa etapa. Refleja el peso del fundador (Elon Musk) y la narrativa de competencia directa con OpenAI.

---

## 6. El efecto OpenAI — un outlier que distorsiona toda la lectura

En el scatter de `monto_ronda` vs `valuacion`, OpenAI aparece completamente aislado en el extremo superior del gráfico. Con USD 300.000M de valuación y USD 6.600M de ronda, comprime al 90% de las startups en un rincón de la pantalla donde no se puede leer nada.

**Al remover OpenAI del gráfico**, la escala cae de 300.000M a 65.000M y aparece la estructura real del mercado:

- **Anthropic** (Serie E) — USD 60.000M de valuación, altísima para una empresa de solo 4 años
- **Databricks** (Serie J) — USD 62.000M, la empresa más madura del dataset
- **Canva** (Serie D) — levantó solo USD 200M pero vale USD 42.000M. Caso de eficiencia de capital extrema
- **Waymo** (Serie B) — levantó USD 5.600M y vale USD 45.000M. Requirió enorme capital por el costo de hardware y regulación

**Conclusión:** un solo outlier puede hacer ilegible un gráfico completo. Identificar y gestionar los outliers antes de visualizar no es opcional — es parte del análisis.

---

## 7. Correlaciones entre variables

| Par de variables | Correlación | Interpretación |
|---|---|---|
| `monto_ronda` ↔ `empleados` | **0,73** (fuerte positiva) | Las empresas que levantan más capital tienden a contratar más |
| `monto_ronda` ↔ `valuacion` | **0,66** (fuerte positiva) | Rondas grandes validan valuaciones altas — se retroalimentan |
| `anio_fundacion` ↔ `empleados` | **-0,52** (moderada negativa) | Las empresas más antiguas tienen más empleados (más negativo = más antiguo) |
| `anio_fundacion` ↔ `valuacion` | **-0,19** (débil) | La antigüedad no predice bien la valuación — los fundadores y la narrativa importan más |

**Hallazgo sobre eficiencia de capital:** la correlación monto/valuación (0,66) no es perfecta. Hay empresas que levantaron mucho y valen poco (baja eficiencia), y empresas que levantaron poco y valen mucho (alta eficiencia). Canva es el caso más extremo del segundo tipo en este dataset.

---

## 8. Startups unicornio

Una startup es unicornio cuando su valuación supera USD 1.000 millones. En este dataset, la mayoría de las empresas con valuación conocida son unicornios — lo cual refleja el sesgo de selección del dataset: es una muestra de las startups de IA más destacadas de 2024-2025, no una muestra aleatoria del mercado.

---

## Síntesis — los 3 hallazgos clave

**1. OpenAI no es una startup — es una categoría aparte.** Con USD 300.000M de valuación, está 5 veces por encima de Anthropic y Databricks, las siguientes en el ranking. Cualquier análisis del ecosistema de IA debe decidir explícitamente si incluir o excluir a OpenAI, porque su presencia distorsiona todos los promedios.

**2. El capital no garantiza valuación — pero la ausencia de capital sí limita.** La correlación monto/valuación es alta (0,66) pero no perfecta. Canva levantó 30 veces menos que Waymo y vale casi lo mismo. El sector y el modelo de negocio importan tanto como el capital levantado.

**3. USA concentra el 71% del ecosistema, pero los outliers más interesantes están fuera.** DeepSeek (China) es autofinanciada y compite con modelos que costaron miles de millones. Mistral (Francia) levantó relativamente poco para el tamaño de su valuación. La geografía del capital no es la geografía de la innovación.
