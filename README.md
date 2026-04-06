# Delivery Promise Optimization Challenge — Mercado Envíos

## Índice
0. [Origen de los datos y construcción del dataset](#0-origen-de-los-datos-y-construcción-del-dataset)
1. [Formalización del problema](#1-formalización-del-problema)
2. [Arquitectura propuesta](#2-arquitectura-propuesta)
3. [Métricas y KPIs](#3-métricas-y-kpis)
4. [Supuestos, riesgos y mitigaciones](#4-supuestos-riesgos-y-mitigaciones)
5. [Problema de ML subyacente](#5-problema-de-ml-subyacente)
6. [Elección de modelos y herramientas](#6-elección-de-modelos-y-herramientas)
7. [Dataset](#7-dataset)
8. [EDA — Análisis Exploratorio](#8-eda--análisis-exploratorio)
9. [Feature Engineering](#9-feature-engineering)
10. [Resultados del modelo principal](#10-resultados-del-modelo-principal)
11. [Feature Importance y Ablation](#11-feature-importance-y-ablation)
12. [Modelo de cooking y notificación al seller](#12-modelo-de-cooking-y-notificación-al-seller)
13. [Instrucciones de reproducción](#13-instrucciones-de-reproducción)
14. [Stack tecnológico y dependencias](#14-stack-tecnológico-y-dependencias)
15. [Uso de herramientas de asistencia generativa](#15-uso-de-herramientas-de-asistencia-generativa)

---

## 0. Origen de los datos y construcción del dataset

### 0.1 Fuentes de datos en BigQuery

El dataset de entrenamiento se construyó a partir de tres tablas del data warehouse de Mercado Envíos en BigQuery, mediante una query que realiza el join, aplica filtros de calidad y clasifica los productos en categorías operativas.

| Tabla | Descripción | Rol en el dataset |
|---|---|---|
| `meli-bi-data.WHOWNER.BT_PROXIMITY_SHIPMENTS_DETAILED` | Tabla principal de envíos de proximidad. Contiene el ciclo de vida completo del envío: timestamps de cada etapa, promesa del modelo actual, tiempos reales, carriers y atributos del pedido. | Base del dataset — una fila por envío |
| `meli-bi-data.WHOWNER.BT_PROXIMITY_ORDERS` | Tabla de ítems por orden. Cada orden puede tener múltiples ítems — se selecciona uno representativo. | Join para obtener el ítem principal de la orden |
| `meli-bi-data.WHOWNER.LK_ITE_ITEMS_MLA` | Lookup de ítems con taxonomía de categorías y vertical de producto. Contiene la jerarquía de categorías (domain_agg2, domain_agg3, category_l3, category_l4). | Clasificación del tipo de producto (`product_type`) |

### 0.2 Construcción del dataset — lógica del pipeline

La query está estructurada en tres CTEs más un join final:

```
BT_PROXIMITY_SHIPMENTS_DETAILED  ──→  shipments CTE
                                          │
BT_PROXIMITY_ORDERS              ──→  order_items CTE (ítem de mayor valor por orden)
                                          │
LK_ITE_ITEMS_MLA                 ──→  item_categories CTE (clasificación product_type)
                                          │
                              JOIN FINAL → dataset_proximity_clean
```

**CTE 1 — `shipments`:** extrae y transforma los atributos del envío. Aplica los siguientes filtros de calidad:

| Filtro | Valor | Motivo |
|---|---|---|
| `CREATED_DATE` | Oct 2025 – Mar 2026 | Ventana temporal del análisis |
| `IS_TEST` | FALSE | Excluir envíos de prueba interna |
| `IS_REPROMISE` | FALSE | Excluir repromesas (1.72% del total) — el modelo predice promesa única |
| `STATUS` | delivered | Solo envíos completados con tiempo real disponible |
| `SIT_SITE_ID` | MLA | Solo Argentina |
| `TOTAL_TIME_REAL` | NOT NULL | Registros con target válido |

En esta CTE también se calculan las features temporales derivadas (`hour_of_day`, `day_of_week`, `is_weekend`, `es_feriado`), se convierten las distancias de metros a kilómetros y se construyen los KPIs de negocio (`is_late`, `promise_error_mins`).

**CTE 2 — `order_items`:** cada orden puede contener múltiples ítems. Para asociar un único tipo de producto a cada envío, se selecciona **el ítem de mayor valor económico** (`TOTAL_AMOUNT DESC`) usando `ROW_NUMBER()`. Esto representa el producto dominante de la orden desde el punto de vista operativo.

**CTE 3 — `item_categories`:** join con la tabla de ítems para obtener la jerarquía de categorías y construir la feature `product_type`. La clasificación se realiza mediante lógica CASE WHEN sobre `domain_agg2`, `domain_agg3`, `category_l3` y `category_l4`.

### 0.3 Clasificación de product_type

La feature `product_type` no existe como columna en el data warehouse — fue construida ad-hoc mediante reglas de negocio sobre la taxonomía de ítems de Mercado Libre. El objetivo es capturar el **tiempo de preparación implícito** de cada tipo de producto, que es la variable operativa más relevante para el lead time.

| Categoría | Descripción operativa | Tiempo de prep. típico |
|---|---|---|
| `helado_postre` | Productos pre-armados o congelados. Mínima preparación. | Bajo |
| `armado_frio` | Sushi, pokes, ensaladas, sandwiches, tablas. Armado sin cocción. | Bajo-medio |
| `desayuno_elaborado` | Cereales, medialunas, tostadas. Preparación simple. | Medio |
| `almacen_empaquetado` | Groceries, snacks, golosinas. Sin preparación. | Muy bajo |
| `bebidas` | Alcohólicas y no alcohólicas. Sin preparación. | Muy bajo |
| `coccion_media` | Platos principales generales, otros frescos. | Medio-alto |
| `coccion_alta_complejidad` | Pizzas, hamburguesas, pastas, carnes, empanadas. Máxima preparación. | Alto |

La lógica de clasificación prioriza `domain_agg3` (nivel de producto) y refina con `category_l3`/`category_l4` para los casos donde el mismo `domain_agg3` contiene productos de preparación muy distinta (ej. `READY-TO-EAT FOOD` incluye tanto sushi como hamburguesas).

Registros con `product_type = NULL` (producto fuera del scope de Food Delivery) fueron excluidos del dataset final mediante el `WHERE ic.product_type IS NOT NULL` del join final.

### 0.4 Sampling representativo

La tabla `dataset_proximity_clean` resultante del pipeline BigQuery contiene el universo completo de envíos filtrados. Para el desarrollo del modelo se extrajo una muestra estratificada por mes:

```sql
-- Sampling: 10,000 registros por mes usando FARM_FINGERPRINT
-- Determinístico, reproducible, sin sesgo de selección
WHERE ABS(MOD(FARM_FINGERPRINT(CAST(ORD_ORDER_ID AS STRING)), 6)) < 1
```

| Mes | Registros |
|---|---|
| Octubre 2025 | 10,000 |
| Noviembre 2025 | 10,000 |
| Diciembre 2025 | 10,000 |
| Enero 2026 | 10,000 |
| Febrero 2026 | 10,000 |
| Marzo 2026 | 10,000 |
| **Total** | **60,000** |

`FARM_FINGERPRINT` garantiza que el sampling es determinístico (misma semilla → mismo resultado) y uniformemente distribuido sobre el espacio de `ORD_ORDER_ID`, sin sesgo temporal ni por carrier.

### 0.5 Variables comentadas en la query (decisiones de exclusión)

| Variable | Motivo de exclusión |
|---|---|
| `SEGMENTO`, `FARMER`, `IS_MELIPLUS` | Atributos del seller no disponibles en todos los contextos de scoring; riesgo de leakage si correlacionan con stores específicos |
| `IS_FREE_SHP` | Análisis EDA confirmó que no aporta señal predictiva sobre el lead time |

### 0.6 Anonimización del dataset para distribución pública

El dataset exportado desde BigQuery contiene identificadores internos de Mercado Libre que no deben publicarse en repositorios públicos. Antes de subir el CSV al repositorio se removieron las siguientes columnas:

| Columna removida | Motivo |
|---|---|
| `ORD_ORDER_ID` | Identificador único de orden — permite rastrear pedidos específicos |
| `ITE_ITEM_ID` | Identificador único de ítem — vinculable a publicaciones de Mercado Libre |

Ninguna de estas columnas es utilizada como feature en los modelos. Su eliminación no afecta la reproducibilidad del entrenamiento ni los resultados.

El archivo disponible en el repositorio es `delivery_promise_challenge_dataset_public.csv` (47 columnas, 60,000 filas). El dataset original de 49 columnas fue generado con el siguiente código:

```python
import pandas as pd

df = pd.read_csv('delivery_promise_challenge_dataset_final.csv')
df_public = df.drop(columns=['ORD_ORDER_ID', 'ITE_ITEM_ID'])
df_public.to_csv('delivery_promise_challenge_dataset_public.csv', index=False)
# Shape resultante: (60000, 47)
```

---

## 1. Formalización del problema

### 1.1 Definición matemática

Dado un pedido de proximidad con features observables en el momento del checkout `t=0`, el sistema debe predecir un intervalo de entrega `[T_low, T_high]` en minutos que se mostrará al comprador.

**Inputs (disponibles en t=0):**

```
x = (distance_km, hour_of_day, day_of_week, is_weekend, es_feriado,
     gmv_order, order_amount, item_count, product_type,
     carrier_name, store_id, city, state, shipping_type)
```

**Outputs:**

```
T_low  = f_q20(x)   → límite inferior de la promesa (percentil 20)
T_high = f_q94(x)   → límite superior de la promesa (percentil 94)
```

**Función objetivo — Pinball Loss:**

La métrica de optimización es la Pinball Loss (también llamada Quantile Loss), que penaliza asimétricamente según el quantil objetivo:

```
L_α(y, ŷ) = α · max(y − ŷ, 0) + (1 − α) · max(ŷ − y, 0)
```

Para `T_low` (`α=0.20`): penaliza más fuertemente subestimar (prometer demasiado rápido).
Para `T_high` (`α=0.94`): penaliza más fuertemente subestimar (prometer un límite superior demasiado bajo → late delivery).

**Restricción operativa principal:**

```
P(T_low ≤ TOTAL_TIME_REAL ≤ T_high) ≥ target_coverage
late_rate = P(TOTAL_TIME_REAL > T_high) ≤ 8.8%  ← referencia del sistema actual
```

### 1.2 Distinción: componente predictivo vs componente de decisión

El problema tiene dos componentes diferenciados:

| Componente | Pregunta | Modelo |
|---|---|---|
| **Predictivo** | ¿Cuánto va a tardar este pedido? | Quantile regression q20 / q94 sobre `TOTAL_TIME_REAL` |
| **De decisión** | ¿Cuándo notificar al seller para que empiece a preparar? | Modelo de cooking + regla de negocio: `t_notify = max(0, T_low − cooking_estimado)` |

El componente de decisión depende del predictivo: usa `T_low` como insumo para calcular el momento óptimo de notificación. Esta separación es intencional — permite iterar cada componente de forma independiente.

---

## 2. Arquitectura propuesta

### 2.1 Diseño end-to-end

```
┌─────────────────────────────────────────────────────────────────┐
│                         OFFLINE (batch)                         │
│                                                                 │
│  BigQuery (raw data)                                            │
│       ↓                                                         │
│  Feature pipeline  ──→  Train / Val / Test split temporal       │
│       ↓                                                         │
│  model_q20   model_q94   model_cooking  ← entrenamiento         │
│       ↓                                                         │
│  Evaluación (pinball loss, late rate, cobertura, MAE)           │
│       ↓                                                         │
│  Artefactos versionados (modelos + encoders + medians)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         ONLINE (serving)                        │
│                                                                 │
│  Checkout event (t=0)                                           │
│       ↓                                                         │
│  Feature extraction en tiempo real                              │
│       ↓                                                         │
│  model_q20.predict(x) → T_low                                   │
│  model_q94.predict(x) → T_high                                  │
│       ↓                                                         │
│  Promesa mostrada al comprador: "Tu pedido llega en X a Y min"  │
│       ↓                                                         │
│  model_cooking.predict(x) → cooking_estimado                    │
│       ↓                                                         │
│  t_notify = max(0, T_low − cooking_estimado)                    │
│       ↓                                                         │
│  Notificación al seller en t=0 + t_notify minutos               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Consideraciones de producción

**Latencia:**
LightGBM tiene latencia de inferencia de microsegundos a milisegundos para una sola predicción. Los tres modelos pueden ejecutarse en secuencia sin impactar el tiempo de respuesta del checkout. No requiere GPU ni infraestructura especial.

**Versionado:**
Cada ciclo de reentrenamiento genera un artefacto versionado que incluye:
- Los tres modelos serializados (`model_q20`, `model_q94`, `model_cooking`)
- Los encoders (`LabelEncoder` por columna, `freq_maps` para STORE_ID y city)
- Las medianas de imputación de train

Todos deben versionarse juntos — un encoder entrenado con un split diferente es incompatible con el modelo correspondiente.

**Monitoreo y drift:**
Se recomienda monitorear las siguientes señales en producción:

| Señal | Frecuencia | Alerta si |
|---|---|---|
| Late rate observado | Diario | > 8.8% por más de 3 días consecutivos |
| Distribución de `TOTAL_TIME_REAL` | Semanal | Shift en mediana > 5 min |
| Distribución de `distance_km` | Mensual | Expansión de cobertura geográfica |
| Cobertura [T_low, T_high] | Semanal | < 70% |
| Mix 2P/3P por zona | Mensual | Cambio > 10pp en alguna zona |

**Reentrenamiento:**
Se recomienda reentrenamiento mensual con ventana deslizante — siempre entrenando en los últimos N meses y evaluando en el mes siguiente. El split temporal debe mantenerse estrictamente.

**Escalabilidad:**
- El modelo actual soporta scoring de millones de pedidos por día sin modificaciones
- La principal limitación de escala es el pipeline de features offline (aggregaciones de STORE_ID) — resoluble con Dataflow o Spark

---

## 3. Métricas y KPIs

### 3.1 Métricas técnicas del modelo

| Métrica | Descripción | Valor obtenido (test) |
|---|---|---|
| Pinball Loss q20 | Error ponderado del límite inferior | **2.2226** |
| Pinball Loss q94 | Error ponderado del límite superior | **2.3530** |
| Cobertura del intervalo | P(T_low ≤ real ≤ T_high) | 72.8% |
| Ancho promedio del intervalo | T_high − T_low | 24.9 min |
| MAE cooking model | Error absoluto medio en estimación de cooking | 5.22 min |

### 3.2 Métricas operativas

| Métrica | Baseline (percentiles fijos) | Modelo final | Mejora |
|---|---|---|---|
| Late rate (real > T_high) | 9.1% | **6.5%** | −2.6pp (−29%) |
| Ancho del intervalo | 27.0 min | **24.9 min** | −2.1 min |
| Pinball q20 | 2.7825 | **2.2226** | −20.1% |
| Pinball q94 | 2.6398 | **2.3530** | −10.9% |

### 3.3 KPI principal de negocio y trade-offs

**KPI principal:** `late_rate` — porcentaje de pedidos que llegan después del `T_high` prometido.

El sistema actual tiene una late rate de referencia del **8.8%**. Reducirla mejora directamente la experiencia del comprador y la reputación del servicio.

**Trade-offs explícitos:**

| Decisión | Beneficio | Costo |
|---|---|---|
| Usar q94 en lugar de q90 | Late rate 6.5% vs 11.1% | Intervalo 4 min más ancho |
| Intervalo más ancho | Menor late rate | Peor UX (incertidumbre mayor para el comprador) |
| Intervalo más estrecho | Mejor UX | Mayor riesgo de late delivery |
| Frequency encoding (vs median) | Sin leakage conceptual | Pérdida de señal → leve degradación de métricas |

La selección de alpha (`q94`) fue calibrada explícitamente para estar por debajo del 8.8% de referencia. No es un valor fijo sino una decisión de negocio que puede ajustarse según el apetito de riesgo del equipo.

---

## 4. Supuestos, riesgos y mitigaciones

### 4.1 Supuestos sobre datos

| Supuesto | Justificación | Riesgo si no se cumple |
|---|---|---|
| `CARRIER_NAME` disponible en t=0 | El carrier se asigna al momento del checkout en el flujo actual | Si la asignación es posterior al checkout, hay leakage → la feature debe removerse |
| Distribución temporal estable | Muestreo estratificado por mes muestra distribuciones similares entre splits | Estacionalidad fuerte (verano/invierno) puede degradar el modelo |
| 10k órdenes/mes representativas | Sampling con FARM_FINGERPRINT es determinístico y sin sesgo | Si hay sesgo en el sampling, las métricas de test no reflejan producción |
| `TOTAL_TIME_REAL` medido correctamente | Variable target sin errores sistemáticos de medición | Errores en timestamps contaminan el target |

### 4.2 Supuestos operativos

| Supuesto | Descripción |
|---|---|
| Promesa única | La promesa se comunica una sola vez en checkout — no se repromesa. Los registros con `IS_REPROMISE=True` fueron excluidos del entrenamiento. |
| Rider disponible en t=T_low | El modelo asume que el rider llega al store aproximadamente en `T_low − cooking_estimado` minutos. Si la asignación de riders falla, el modelo de notificación pierde validez. |
| Stores conocidos dominan el volumen | El 81.6% de los stores en test tienen historial en train. El 18.4% restante (cold start) recibe frequency=0. |

### 4.3 Supuestos sobre estabilidad temporal

| Supuesto | Indicador de violación | Mitigación |
|---|---|---|
| La relación distance_km → lead time es estable | Cambio en distribución de `transit_time_real` | Monitoreo semanal de la distribución |
| El mix 2P/3P es estable por zona | Variación >10pp en el mix mensual por ciudad | Reentrenamiento ante expansión geográfica |
| Los product_types no cambian | Aparición de nuevas categorías | Pipeline de encoding con manejo de categorías desconocidas |
| El comportamiento de stores es estable | Apertura/cierre masivo de stores | Actualización de frequency maps en reentrenamiento |

### 4.4 Riesgos principales

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Concept drift en velocidad de stores | Media | Alto | Reentrenamiento mensual + monitoreo de late rate |
| Expansión geográfica fuera del dominio de entrenamiento | Media | Alto | Retrain con datos de nueva zona antes del lanzamiento |
| Cold start en stores nuevos | Alta (18% en test) | Medio | Usar mediana global como fallback; agregar features de características del store |
| Degradación del cooking model (MAE ~5 min) | Baja | Medio | Monitorear MAE en producción con feedback de `cooking_time_real` real |

---

## 5. Problema de ML subyacente

### 5.1 Tipo de problema

**Quantile Regression** — estimación de percentiles condicionales de una variable continua (`TOTAL_TIME_REAL`) dado un vector de features observables en t=0.

No es un problema de regresión estándar (que predice la media) ni de clasificación. Se necesita modelar explícitamente la incertidumbre de la distribución condicional — cuánto puede tardar un pedido en el peor caso (T_high) y en el mejor caso (T_low).

### 5.2 ¿Por qué ML y no un enfoque heurístico?

| Enfoque | Descripción | Limitación |
|---|---|---|
| **Percentiles globales fijos** (baseline) | T_low = p20 global, T_high = p94 global | No captura variabilidad por store, distancia, hora ni producto. Late rate: 9.1% |
| **Percentiles por segmento** | Calcular percentiles por `product_type × hora` | Explota rápidamente en combinaciones escasas. No generaliza a nuevos segmentos |
| **Reglas de negocio manuales** | "Helados: 15-40 min; cocción alta: 25-55 min" | No escala, no se adapta, requiere mantenimiento manual constante |
| **ML — Quantile Regression** | Modelo que aprende la distribución condicional completa | Captura interacciones complejas entre features. Generaliza. Reentrenable. |

El modelo ML logró **−20.1% en Pinball q20** y **−2.6pp en late rate** sobre el baseline de percentiles globales. La mejora proviene de capturar que pedidos con `distance_km` alta, en hora pico, en stores con bajo volumen, tienen distribuciones de lead time estructuralmente distintas.

### 5.3 Justificación de quantile regression vs prediction intervals

Se optó por quantile regression directa (no por intervalos de predicción bootstrapeados ni por modelos probabilísticos como NGBoost) porque:
- La Pinball Loss es la métrica del enunciado — optimizarla directamente es más coherente que aproximarla
- LightGBM con `objective='quantile'` es computacionalmente eficiente y bien calibrado en la práctica
- Permite control explícito sobre cada quantil de forma independiente

---

## 6. Elección de modelos y herramientas

### 6.1 LightGBM — justificación técnica

| Criterio | Justificación |
|---|---|
| **Soporte nativo de quantile loss** | `objective='quantile'` con `alpha` configurable — sin necesidad de wrappers |
| **Eficiencia en datos tabulares** | Supera a redes neuronales en datasets < 500k filas con features heterogéneas |
| **Manejo de variables mixtas** | Features numéricas, binarias y categóricas (post-encoding) en un mismo modelo |
| **Velocidad de inferencia** | Microsegundos por predicción — viable para serving en tiempo real en checkout |
| **Interpretabilidad** | Feature importance (gain) directamente accesible — útil para análisis de errores y defensa |

**Alternativas evaluadas y descartadas:**

| Modelo | Motivo de descarte |
|---|---|
| XGBoost | Soporte de quantile loss menos maduro que LightGBM en la versión evaluada |
| NGBoost | Modela la distribución completa (no solo quantiles), mayor complejidad sin beneficio claro |
| CatBoost | Requiere más tuning para categoricals de alta cardinalidad; menor familiaridad del equipo |
| Redes neuronales | Overhead de arquitectura innecesario para este tamaño de dataset; peor interpretabilidad |

### 6.2 Esquema de entrenamiento y validación

```
Split temporal estricto (no random):
├── Train:  Oct 2025 – Ene 2026  (39,995 filas, 66.7%)
├── Val:    Feb 2026              ( 9,999 filas, 16.7%)  ← tuning de Optuna
└── Test:   Mar 2026              ( 9,996 filas, 16.7%)  ← evaluación final única
```

**¿Por qué split temporal y no cross-validation?**
La cross-validation aleatoria contaminaría el modelo con datos del futuro — un pedido de marzo podría aparecer en train mientras uno de octubre en test. En series temporales, el modelo siempre debe entrenarse en pasado y evaluarse en futuro.

**Optimización de hiperparámetros (Optuna):**
50 trials de búsqueda bayesiana sobre val. El test nunca fue utilizado durante la optimización — se evaluó una única vez al final.

### 6.3 Escenarios adversos identificados

| Escenario | Descripción | Impacto | Mitigación |
|---|---|---|---|
| **Drift de distribución** | Cambio en patrones de demanda (ej. nuevo barrio, nueva flota) | Degradación gradual de late rate | Monitoreo de late rate observado + reentrenamiento mensual |
| **Cold start de stores** | 18.4% de stores en test sin historial (frequency=0) | Predicciones menos precisas para stores nuevos | Frequency=0 como fallback explícito; agregar features de características del store |
| **Sobreajuste en Optuna** | 50 trials sobre val puede ajustar ruido de val | Brecha val-test en métricas | Early stopping en entrenamiento; evaluación final solo en test |
| **Mala calibración en colas** | p94 puede estar mal calibrado en zonas escasas | Late rate mayor en segmentos pequeños | Análisis de late rate segmentado por product_type y zona |
| **Cambio en mix 2P/3P** | Expansión de un tipo de logística cambia la distribución | Sesgo sistemático en predicciones | SHIPPING_TYPE como feature explícita; monitoreo del mix mensual |
| **Desbalance temporal** | Eventos excepcionales (ej. Black Friday) no representados | Picos de late rate en fechas especiales | Agregar features de calendario + reentrenamiento post-evento |

---

## 7. Dataset

| Atributo | Valor |
|---|---|
| Registros totales | 60,000 |
| Período | Oct 2025 – Mar 2026 (6 meses) |
| Muestra por mes | 10,000 registros |
| Sampling | Estratificado por mes (FARM_FINGERPRINT) |
| Columnas originales | 49 |
| Columnas usadas en modelo | 14 features + 1 target |

### Variables excluidas por data leakage

| Columna | Motivo |
|---|---|
| `TOTAL_TIME_PROMISE`, `COOKING_TIME_PROMISE`, `TRANSIT_TIME_PROMISE` | Output del modelo actual |
| `cooking_time_real`, `transit_time_real` | Valores futuros |
| `t1` a `t5` checkpoints | Eventos futuros |
| `mins_rider_waits_in_store` | Evento futuro |
| `READY_TO_COOK_DT` | Calculado por el sistema actual |
| `IS_REPROMISE` | Registros excluidos — promesa única por enunciado |

---

## 8. EDA — Análisis Exploratorio

### 8.1 Calidad del dataset

| Métrica | Valor |
|---|---|
| Registros | 60,000 |
| Duplicados | 0 |
| Nulos en `TOTAL_TIME_REAL` | 0 |
| Nulos en `cooking_time_real` | 983 (1.64%) |
| Nulos en `SHP_BUYER_CITY_NAME` | 3 (0.00%) |

### 8.2 Distribución del target (TOTAL_TIME_REAL)

| Percentil | Minutos |
|---|---|
| p20 | 20 min |
| p50 (mediana) | 28 min |
| p80 | 39 min |
| p94 | 47 min |
| p99 | 71 min |
| máximo | 1,652 min (outlier) |

- Media: 30.1 min | Desvío estándar: 22.2 min
- Valores > 180 min: 10 registros → removidos con filtro aplicado en Feature Engineering
- **Tasa global de late (is_late):** 8.8%

### 8.3 Componentes del lead time

| Componente | p20 | p50 | p80 | Correlación con target |
|---|---|---|---|---|
| `cooking_time_real` | 9.0 min | 14.0 min | 21.0 min | **0.757** |
| `transit_time_real` | 5.1 min | 8.3 min | 12.6 min | 0.524 |
| `distance_km` | — | 1.34 km | — | 0.367 |

### 8.4 Patrones temporales

- **Horas de mayor demora:** 8-10 hs (media 32-33 min) y 20-21 hs (media 30-33 min, mayor volumen)
- **Feriados vs días normales:** 31.8 vs 30.1 min (+5.6%) — diferencia menor a lo esperado

### 8.5 Lead time por product_type

| Producto | Media | Mediana | Volumen |
|---|---|---|---|
| helado_postre | 25.0 min | 23.0 min | 11,154 |
| desayuno_elaborado | 29.2 min | 27.0 min | 2,173 |
| coccion_alta_complejidad | 30.5 min | 28.0 min | 29,774 |
| bebidas | 30.9 min | 29.0 min | 939 |
| almacen_empaquetado | 30.6 min | 28.0 min | 519 |
| coccion_media | 31.6 min | 29.0 min | 5,290 |
| armado_frio | 34.0 min | 32.0 min | 10,151 |

### 8.6 Lead time por SHIPPING_TYPE

| Tipo | Descripción | Media | Mediana | p94 |
|---|---|---|---|---|
| 2P | Vendedor maneja logística | 37.9 min | 35.0 min | 69.0 min |
| 3P | Carrier maneja logística | 29.8 min | 28.0 min | 51.0 min |

**2P es 7.1 min más lento en mediana que 3P.** La diferencia es consistente en todos los product_types.

### 8.7 Cardinalidad

| Variable | Categorías únicas |
|---|---|
| `STORE_ID` | 5,349 |
| `SHP_BUYER_CITY_NAME` | 303 |
| `CARRIER_NAME` | 72 |
| `SHP_SELLER_STATE_NAME` | 4 |
| `SHIPPING_TYPE` | 2 (2P / 3P) |
| `product_type` | 7 |

---

## 9. Feature Engineering

### 9.1 Filtro del target

```python
df_model = df[df['TOTAL_TIME_REAL'] <= 180]
# 59,990 registros (removidos: 10 outliers extremos)
```

### 9.2 Split temporal

| Split | Período | Filas | % |
|---|---|---|---|
| Train | Oct 2025 – Ene 2026 | 39,995 | 66.7% |
| Validation | Feb 2026 | 9,999 | 16.7% |
| Test | Mar 2026 | 9,996 | 16.7% |

### 9.3 Estrategia de encoding

| Variable | Método | Justificación |
|---|---|---|
| `product_type` | Label encoding | 7 categorías, baja cardinalidad |
| `SHP_SELLER_STATE_NAME` | Label encoding | 4 categorías |
| `SHIPPING_TYPE` | Label encoding | 2 categorías (2P / 3P) |
| `STORE_ID` | **Frequency encoding** | Alta cardinalidad (5,349) — conteo de órdenes en train como proxy de volumen operativo |
| `SHP_BUYER_CITY_NAME` | **Frequency encoding** | Alta cardinalidad (303) — mismo criterio |
| `CARRIER_NAME` | Label encoding | 72 carriers |

**Decisión clave:** se descartó median encoding para STORE_ID porque implica usar `TOTAL_TIME_REAL` como base de cálculo, lo que supone disponibilidad de aggregaciones históricas actualizadas en producción — un supuesto no garantizado. Frequency encoding captura el volumen operativo del store sin depender del target.

Cold start: stores sin historial en train → frequency=0.
- Nuevos en val: 713 stores | en test: 984 stores

### 9.4 Features finales (14)

```
distance_km, hour_of_day, day_of_week, is_weekend, es_feriado,
gmv_order, order_amount, ITEM_COUNT,
product_type, CARRIER_NAME, STORE_ID,
SHP_BUYER_CITY_NAME, SHP_SELLER_STATE_NAME, SHIPPING_TYPE
```

---

## 10. Resultados del modelo principal

### 10.1 Comparación baseline vs modelo final (test set)

| Modelo | Pinball q20 | Pinball q94 | Cobertura | Ancho | Late rate |
|---|---|---|---|---|---|
| Baseline (percentiles fijos) | 2.7825 | 2.6398 | 75.1% | 27.0 min | 9.1% |
| LGBM final (q20/q94) | **2.2226** | **2.3530** | 72.8% | **24.9 min** | **6.5%** |
| Mejora | **−20.1%** | **−10.9%** | −2.3pp | **−2.1 min** | **−2.6pp** |

### 10.2 Distribución del intervalo en test

| | T_low | T_high |
|---|---|---|
| Media | 22.0 min | 46.8 min |
| Mediana | 21.4 min | 46.5 min |
| Ancho promedio | — | **24.9 min** |

### 10.3 Selección de alpha para T_high

| Alpha | Late rate | Ancho intervalo |
|---|---|---|
| q90 | 11.1% | 19.7 min |
| q92 | 9.0% | 21.5 min |
| **q94** | **7.0%** | **23.7 min** ← seleccionado |
| q95 | 5.8% | 25.2 min |

---

## 11. Feature Importance y Ablation

### 11.1 Feature importance (gain %)

| Feature | q20 | q94 |
|---|---|---|
| distance_km | 28.3% | 23.4% |
| STORE_ID | 13.7% | 14.8% |
| order_amount | 12.0% | 10.7% |
| gmv_order | 10.7% | 9.1% |
| SHP_BUYER_CITY_NAME | 9.3% | 9.8% |
| hour_of_day | 6.4% | 8.1% |
| product_type | 7.6% | 6.2% |
| CARRIER_NAME | 3.1% | 5.4% |
| SHIPPING_TYPE | 0.1% | 2.4% |
| es_feriado | 0.2% | 0.5% |

`SHIPPING_TYPE` tiene mayor impacto en q94 (2.4%) que en q20 (0.1%) — el tipo de logística afecta principalmente los casos extremos (cola superior), no los pedidos rápidos.

### 11.2 Ablation STORE_ID

| | Pinball q94 | Late rate | Ancho |
|---|---|---|---|
| Con STORE_ID | 2.3530 | **6.5%** | 24.9 min |
| Sin STORE_ID | 2.3241 | 7.2% | 24.0 min |

Sin STORE_ID el Pinball q94 mejora levemente pero la late rate empeora (+0.7pp). Se mantiene STORE_ID porque el objetivo de negocio es reducir late rate.

---

## 12. Modelo de cooking y notificación al seller

### 12.1 Objetivo y fórmula

```
t_notify = t_checkout + max(0, T_low − cooking_estimado)
```

### 12.2 Resultados del cooking model

| Métrica | Val | Test |
|---|---|---|
| MAE | 5.28 min | **5.22 min** |
| Mediana real `cooking_time_real` | 14.0 min | 14.0 min |
| Mediana predicha | 14.0 min | 14.0 min |

El modelo está bien calibrado en la mediana (sin bias sistemático).

### 12.3 Distribución de notificación al seller

| Momento | % pedidos |
|---|---|
| 0 min (inmediato) | 1.6% |
| 1 – 5 min post-checkout | 19.6% |
| 5 – 10 min post-checkout | 40.9% |
| > 10 min post-checkout | 37.9% |
| **Mediana** | **8.4 min** |

### 12.4 Promesa y notificación por producto

| Producto | T_low | T_high | Cooking est. | Notif. en |
|---|---|---|---|---|
| helado_postre | 17m | 40m | 11m | 5m post-checkout |
| coccion_alta_complejidad | 22m | 46m | 13m | 9m post-checkout |
| desayuno_elaborado | 21m | 45m | 12m | 9m post-checkout |
| almacen_empaquetado | 23m | 49m | 12m | 10m post-checkout |
| armado_frio | 24m | 52m | 13m | 11m post-checkout |
| coccion_media | 23m | 48m | 13m | 10m post-checkout |
| bebidas | 22m | 49m | 13m | 9m post-checkout |

---

## 13. Instrucciones de reproducción

### 13.1 Requisitos previos

- Cuenta de Google con acceso a Google Colab
- No se requiere configuración adicional — el dataset se descarga automáticamente desde el repositorio

### 13.2 Pasos para reproducir

1. Abrir el notebook en Google Colab:
   - Ir a [colab.research.google.com](https://colab.research.google.com)
   - `Archivo → Abrir notebook → GitHub`
   - Pegar la URL del repositorio: `https://github.com/nicogblanc/challenge_tecnico_delivery_promise_optimization`
   - Seleccionar `delivery_promise_challenge_v2.ipynb`

2. Ejecutar todas las celdas en orden:
   - `Entorno de ejecución → Ejecutar todo` (o `Ctrl+F9`)
   - La primera celda descarga el dataset automáticamente desde el repositorio — no requiere ningún paso manual adicional

3. El notebook ejecuta las siguientes etapas en secuencia:
   ```
   Celda 0   → Carga del dataset desde Drive
   Celdas 1-7 → EDA completo
   Celda 8   → Análisis de cardinalidad
   Celdas 9-11 → Feature Engineering (filtro, split, encoding)
   Celda 12  → Definición de quantiles
   Celdas 13-19 → Entrenamiento de modelos (baseline → LGBM → Optuna → selección alpha)
   Celda 20  → Evaluación final en test
   Celda 21  → Feature importance
   Celda 22  → Ablation STORE_ID
   Celdas 23-25 → Modelo cooking y notificación al seller
   ```

4. **Tiempo estimado de ejecución completa:** 15-25 minutos (dominado por Optuna: 50 trials × 2 modelos)

### 13.3 Notas de reproducibilidad

- Optuna con `n_trials=50` tiene variabilidad entre ejecuciones — los hiperparámetros exactos pueden diferir levemente, pero las métricas finales deben ser similares (±0.01 en Pinball Loss)
- `random_state=42` fijado en todos los modelos que lo soportan
- El split temporal es determinístico (no depende de random state)

---

## 14. Stack tecnológico y dependencias

### 14.1 Entorno de ejecución

| Componente | Versión |
|---|---|
| Entorno | Google Colab (Python 3.12) |
| Sistema operativo | Linux (Ubuntu, Colab runtime) |

### 14.2 Dependencias principales

| Librería | Uso en el proyecto |
|---|---|
| `pandas` | Manipulación de datos, encoding, splits |
| `numpy` | Operaciones numéricas, cálculo de métricas |
| `lightgbm` | Modelos de quantile regression y MAE |
| `scikit-learn` | LabelEncoder, métricas auxiliares |
| `optuna` | Optimización bayesiana de hiperparámetros |
| `matplotlib` | Visualizaciones del EDA |
| `seaborn` | Visualizaciones estadísticas del EDA |

### 14.3 Versiones (Google Colab, abril 2026)

```
pandas        >= 2.0
numpy         >= 1.24
lightgbm      >= 4.0
scikit-learn  >= 1.3
optuna        >= 3.0
matplotlib    >= 3.7
seaborn       >= 0.12
```

Para verificar las versiones exactas del entorno de ejecución, correr en Colab:
```python
import pandas, numpy, lightgbm, sklearn, optuna, matplotlib, seaborn
for lib in [pandas, numpy, lightgbm, sklearn, optuna, matplotlib, seaborn]:
    print(f"{lib.__name__}: {lib.__version__}")
```

---

## 15. Uso de herramientas de asistencia generativa

### 15.1 Herramienta utilizada

**Claude Code** (Anthropic) — asistente de programación con contexto de conversación persistente, utilizado a lo largo de todo el desarrollo del proyecto.

### 15.2 Rol de la herramienta en el flujo de trabajo

El desarrollo siguió un flujo de **pair programming asistido**: el candidato condujo todas las decisiones técnicas y de negocio, mientras Claude Code asistió en implementación, explicación de conceptos y generación de código.

**El candidato tomó las siguientes decisiones de forma autónoma:**
- Selección de features y exclusión por data leakage
- Elección de frequency encoding sobre median encoding (y justificación conceptual)
- Definición del intervalo [T_low = q20, T_high = q94] y calibración del alpha
- Diseño del modelo de notificación al seller y su fórmula
- Validación de todos los resultados ejecutando el código en Colab
- Identificación de bugs (dtype errors, shape mismatches) y decisión sobre cómo resolverlos
- Revisión crítica de los outputs y detección de comportamientos inesperados

**Claude Code asistió en:**
- Generación de código Python para EDA, encoding, entrenamiento y evaluación
- Explicación de conceptos (pinball loss, quantile regression, temporal split)
- Debugging de errores de ejecución
- Redacción del README y documentación técnica
- Revisión del notebook y corrección de inconsistencias

### 15.3 Registro de interacción

El historial completo de la conversación de desarrollo está disponible como log adjunto al repositorio. Este log refleja el proceso iterativo real: el candidato ejecutaba el código, compartía los resultados, y en base a esos resultados tomaba decisiones sobre los próximos pasos.

El log evidencia que el flujo fue exploratorio y basado en resultados reales — no una generación automatizada de código sin comprensión del problema.

### 15.4 Postura sobre el uso de IA en el contexto del rol

El uso de herramientas de asistencia generativa es consistente con el perfil del rol DS Engineer en producción: la capacidad de integrar estas herramientas de forma efectiva, validar sus outputs críticamente y mantener el criterio técnico propio es una competencia relevante. El objetivo no fue delegar el pensamiento, sino acelerar la implementación sin sacrificar el rigor metodológico.
