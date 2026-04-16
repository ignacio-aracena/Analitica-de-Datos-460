# CLAUDE.md
Guía operativa para este repo. Se carga en cada sesión — mantenerlo corto.

## Qué es este repo

Materiales de las **clases tutoriales** de **Analítica de Datos**, Licenciatura en Negocios Digitales, Universidad de San Andrés, Otoño 2026.

**Roles:**
- **Pablo Sciolla** — profesor titular. Dicta las magistrales. Sus PDFs definen qué conceptos vieron los alumnos antes de cada tutorial.
- **Ignacio Aracena + Juan Costa** — profesores de tutoriales, dueños del repo. Los tutoriales aplican la teoría a casos de negocio reales con notebooks.

**Audiencia:** alumnos de negocios, no ingenieros. Argentinos. Usan IA (Claude) para escribir código — no aprenden a programar. El foco es interpretar outputs y tomar decisiones de negocio.

## Flujo de trabajo para clases nuevas

Toda clase nueva pasa por dos etapas en orden:

1. **/think en Claude.ai** — el profesor diseña el caso antes de tocar código: industria, empresa, problema de negocio, qué entra en clase, qué va para casa. Solo con el caso aprobado se pasa al siguiente paso.
2. **Ejecución en Claude Code** — genera los tres archivos de la clase siguiendo este archivo.

## Estructura de cada clase

Cada `Clase NN - <tema>/` contiene:
- `practicaNN.ipynb` — notebook del tutorial (naming estándar desde Clase 04)
- `<dataset>.csv` — datos del caso de negocio
- `hallazgos.md` — informe de negocio: qué encontramos, qué decidimos, por qué. Lo que estudian para el parcial.
- `documentacion_tecnica.md` — explicación técnica celda por celda. Para quien quiere entender el código.

El PDF del tutorial de Juan define el diseño pedagógico. El PDF de Pablo define qué ya saben los alumnos y no hay que re-explicar.

## Cómo ejecutar y validar notebooks

```bash
/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute <archivo.ipynb> --output <archivo.ipynb>
```

El entorno con todo el stack instalado es `/opt/anaconda3/bin/python` (pandas 2.3, numpy 1.26, sklearn 1.5, seaborn 0.13, matplotlib 3.10).

## Reglas no negociables

**Código:**
- Nivel accesible para alumnos de negocios. Si un alumno sin experiencia en Python no puede leer una línea y entender grosso modo qué hace, hay que simplificarla.
- Funciones de formato definidas con `def`, nunca `lambda`
- Leyendas con `ax.legend()` estándar
- Fronteras de clasificación con `DecisionBoundaryDisplay.from_estimator()`
- Sin `hasattr`, sin list comprehensions anidadas

**Idioma:** español rioplatense en todo — notebooks, docs, comentarios, variables cuando ayuda a la lectura.

**Notebooks:** estilo sparse — solo código + observaciones cortas de output. La explicación vive en `documentacion_tecnica.md`. Sin emojis. Sin párrafos largos dentro del notebook.

**Git:** no commitear sin que el profesor lo pida explícitamente. Dejar cambios staged y avisar.

**Datasets sintéticos:** no van los scripts generadores. Solo el CSV final en el repo.

## Referencia rápida de módulos

La materia tiene 4 módulos (ver programa oficial):
- **Módulo 1** — EDA, estadística descriptiva, visualización, storytelling
- **Módulo 2** — ML Supervisado: regresión lineal/logística, KNN, árboles, RF, GB, Naive Bayes. Evaluación técnica, económica y estratégica. **Examen Parcial al cierre.**
- **Módulo 3** — ML No Supervisado: clustering, reducción de dimensionalidad
- **Módulo 4** — Deep Learning. **Examen Final al cierre.**

Evaluación: 20% quiz semanal + 20% conceptual + 30% Parcial 1 + 30% Parcial 2.
