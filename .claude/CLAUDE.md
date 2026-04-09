# CLAUDE.md

Guía operativa para asistir en este repositorio. Leer antes de proponer cambios.

## Qué es este repo

Materiales de las **clases tutoriales** de **Analítica de Datos**, Licenciatura en Negocios Digitales, **Universidad de San Andrés** (UdeSA), Otoño 2026.

- **Profesor titular (teóricas / magistrales):** Pablo Sciolla — arma los PDFs de cada clase con la teoría.
- **Profesores de tutoriales (dueños del repo):** Ignacio Aracena y Juan Costa — los tutoriales **revisitan** brevemente la teoría y bajan a la práctica con notebooks.
- **Audiencia:** estudiantes de **negocio**, no de ingeniería. Argentinos, español rioplatense.

El repo se sube a GitHub: los alumnos lo clonan para tener los notebooks y datasets.

## Filosofía pedagógica (no negociable)

> Los alumnos **no aprenden a programar Python**. Aprenden a **usar herramientas de IA (Claude, etc.)** para limpiar datos, modelar e interpretar resultados en clave de negocio.

Implicancias para todo material que se arme:

- Cada paso de código va acompañado de markdown corto con el esquema **Qué vemos / Qué hacemos / Por qué (negocio)**.
- Priorizar **interpretación** de resultados sobre detalles de sintaxis.
- Preferir contextos de **negocio digital** (marketing, e-commerce, SaaS) antes que ejemplos genéricos.
- Cuando una clase pide "10 minutos exactos", contar celdas — no inflar.
- Técnicas avanzadas se mencionan como gancho de "qué sigue", no se incluyen en el flujo principal.

## Programa de la materia (4 módulos)

Está en `Programa Analítica de Datos - Otoño 2026.pdf` (fuera del repo, en local del profesor). Resumen:

| Módulo | Tema | Estado |
|---|---|---|
| **1** | La analítica de datos y las organizaciones (problema de negocio, EDA, estadística descriptiva, storytelling, viz) | En curso (Clases 01-04) |
| **2** | Aprendizaje Automático Supervisado (Regresión Lineal, Logística, KNN, Árboles, RF, GB, Naïve Bayes; evaluación técnica + económica + estratégica) | Empieza en Clase 04 (intro) |
| **3** | Aprendizaje No Supervisado y Reducción de Dimensionalidad (K-Means, Hierarchical, DBSCAN, GMM, PCA, t-SNE, UMAP) | Pendiente |
| **4** | Introducción a Deep Learning (ANNs, CNN, RNN, Transformers / LLMs) | Pendiente |

**Examen Parcial** entre Módulos 2 y 3. **Examen Final** al cierre del Módulo 4.
**Evaluación:** 20% quiz semanal + 20% conceptual teóricas/tutoriales + 30% Parcial 1 + 30% Parcial 2.

## Mapeo Clases ↔ Módulos

| Clase | Carpeta | Módulo | Foco | Dataset |
|---|---|---|---|---|
| 01 | `Clase 01 - Data Visualization/` | 1 | Repaso Python, Pandas, Matplotlib/Seaborn | `startups.csv` |
| 02 | `Clase 02 - Data Profiling & Visualization/` | 1 | Data profiling + viz | `Superstore.csv` |
| 03 | `Clase 03 - Tableau/` | 1 | Tableau (sin notebook, solo dataset) | `Superstore.csv` |
| 04 | `Clase 04 - Data Cleaning & Linear Regression/` | 1 → 2 | Data cleaning + intro ML (regresión lineal simple). **Primera clase con el patrón notebook minimal + 2 docs.** | `campañas_marketing.csv` |

## Convenciones por carpeta de Clase

Cada `Clase NN/` con notebook contiene típicamente **5 archivos**:

- **PDF teórico** del profesor titular (`Clase N - <tema>.pdf`) — **no editar**, es la fuente de verdad de los temas a cubrir en el tutorial.
- **Notebook del tutorial**. Naming inconsistente entre clases: en 01-02 se usa `practicaNN.ipynb`, en 04 se adoptó `Clase04_<tema>.ipynb`. **Respetar el existente, no renombrar sin pedir.**
- **Dataset(s)** CSV.
- **`documentacion_tecnica.md`** — explicación celda por celda del notebook (mecánica del código). Para el alumno que quiere entender qué hace cada línea de Python.
- **`hallazgos.md`** — informe analítico de negocio: qué encontramos en los datos, qué decisiones tomamos y por qué. Para el alumno que estudia para el parcial o quiere ver el "qué" en lugar del "cómo".

**Excepción:** Clase 03 (Tableau) solo tiene el dataset — no hay notebook ni docs porque toda la práctica se hace en Tableau.

**Patrón triple-archivo (notebook + 2 docs)** establecido en Clase 04, retroactivamente cubierto en 01 y 02. Cuando armes una clase nueva, generá los 3 archivos.

> Cuando armes un notebook nuevo: **el PDF de la clase manda**. Antes de proponer contenido, leer el PDF y validar que cubrís todos los conceptos teóricos centrales (no necesariamente todos los detalles, pero sí los conceptos que quedan en pizarra).

## Convenciones de los notebooks

**Estilo "minimal" (estándar a partir de Clase 04):** los notebooks son intencionalmente sparse — solo código + observaciones cortas de output. La explicación detallada vive en los `.md` complementarios.

- **Idioma:** español (Argentina). Nombres de variables/funciones en español cuando ayuda a leer.
- **Sin emojis** en headers ni en celdas. Tono profesional.
- **Markdown sparse**: solo headers de sección (`## A.1`, `## B.3`, etc.) con 1-2 líneas máximo de contexto. **Nada de párrafos largos explicando qué hace cada celda** — eso va en `documentacion_tecnica.md`.
- **Observaciones de output**: una línea markdown corta debajo de los outputs clave, del estilo `→ R² ≈ 0.95 → la inversión explica el 95% de las ventas`. Solo donde el output requiere lectura, no después de cada celda.
- **Contexto de negocio al inicio**: empezar con "Somos el equipo de datos de…" o similar antes de cualquier código. Esto sí queda en el notebook (1 párrafo corto).
- **Sección "Para ver en casa" al final**: contenido teórico del PDF que no entra por tiempo en clase pero **es evaluable**. Pattern establecido en Clase 04 — usarlo cuando haya conceptos importantes del PDF que no entran en el budget de minutos.
- **Print intermedios** para que el alumno vea el efecto de cada paso.
- **Imputación**: preferir mediana sobre media (robusta a outliers).
- **`sns.set_style("whitegrid")`** como estilo por defecto.

**El "hilo" para no programadores**: incluso en estilo sparse, las clases siguen siendo para alumnos de negocio. Los headers de sección (`## A.1`, `## A.2`, etc.) y las observaciones de output cortas son lo que les permite seguir el flujo sin entender cada línea de código.

## Stack y entorno Python

- `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`
- `scikit-learn` para modelado (`train_test_split`, `LinearRegression`, `metrics`, etc.)
- Evitar `statsmodels` salvo mención opcional al final
- **Importante**: el sistema tiene Python 3.14 en `/opt/homebrew/bin/python3` **sin** pandas/sklearn. El intérprete con todo el stack instalado es **`/opt/anaconda3/bin/python`** (pandas 2.3, numpy 1.26, sklearn 1.5, seaborn 0.13, matplotlib 3.10, jupyter).
- Para ejecutar/validar notebooks: `/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute <archivo.ipynb> --output <archivo.ipynb>`

## Datasets

- Algunos son reales/descargados (Superstore, startups). Otros son **sintéticos generados ad hoc** para la clase (ej. el de Clase 04 — campañas de marketing digital con relación oculta `ventas ≈ 3.2 · inversión + ruido`).
- **Los scripts generadores de datasets sintéticos NO van en el repo** (los alumnos se confundirían). Si necesitás regenerar, hacelo con un script temporal fuera del repo, copiá el CSV final y borrá el script.
- Naming actual del dataset de Clase 04: `campañas_marketing.csv` (sin "_crudo") — el profesor renombró deliberadamente.

## Git / GitHub

- Branch principal: `main`. PRs contra `main`.
- El repo se publica en GitHub y los alumnos lo clonan — **cualquier cosa que se commitee es visible para los estudiantes**. Pensar dos veces antes de subir scripts internos, soluciones, datasets crudos de fabricación, etc.
- **No commitear sin que el usuario lo pida explícitamente.** Default: dejar los cambios staged y avisar.
- No `--force-push` a `main`.

## Cuando arme contenido nuevo

1. **Leer el PDF de la clase primero** (el de Pablo) y mapear todos los conceptos centrales.
2. Verificar contra esta lista que el notebook los cubre. Los conceptos del PDF que **no entren por tiempo** se mencionan al menos como markdown corto, no se omiten en silencio.
3. Respetar el tope de tiempo si el profesor lo dice ("10 minutos exactos" → ~14-16 celdas máximo).
4. Validar que el notebook **corre de punta a punta sin errores** con `nbconvert --execute` (anaconda).
5. Confirmar con el profesor antes de borrar archivos o renombrar cosas existentes.

## Memoria persistente

Hay memorias del usuario y del proyecto en `~/.claude/projects/<este-repo>/memory/`. Antes de empezar tareas no triviales, conviene revisarlas — sobre todo `feedback_*.md` para no repetir correcciones pasadas.
