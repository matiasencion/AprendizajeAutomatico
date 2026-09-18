# Experimento: atributo de paridad (`match_evenness`)

## 2026-09-16 — Efecto de un atributo de "paridad" sobre Bayes e ID3

Este documento registra un experimento controlado para evaluar si un nuevo
atributo, pensado específicamente para ayudar a detectar empates, mejora
las métricas de Naive Bayes y del árbol de decisión ID3. **No se utilizó en
ningún momento el conjunto de test 2024-2025**; toda la validación se hizo
sobre `train` (partidos anteriores a 2024-01-01), reutilizando los mismos
folds temporales anuales que ya usan `bayesNotebook.ipynb` y
`treeNotebook.ipynb` (validar cada año 2013-2022 entrenando con los 5 años
anteriores).

## Motivación

Los atributos existentes (`record_difference`, `last_matches_difference`,
`goal_difference_value`, `attack_difference`, `defense_difference`,
`elo_difference`, `h2h_difference`) son todos **direccionales**: indican
*quién* es mejor (local o visitante) y cuánto. Ninguno indica directamente
*qué tan pareja* es la diferencia de nivel entre los equipos,
independientemente de quién sea mejor — que es la señal más directa para
un empate. Además, están fuertemente correlacionados entre sí (correlación
de Pearson entre 0.67 y 0.85 con `elo_difference` en `train`), por lo que
agregar uno nuevo de este tipo aporta poca información marginal, y para
Naive Bayes en particular puede ser contraproducente por la violación del
supuesto de independencia condicional.

## Implementación

- **`load.py`**: se agregó `elo_closeness = abs(elo_difference)` (calculado
  con el mismo ELO secuencial y causal que ya existía; no se agregó
  ninguna fuente de información nueva, solo se reexpresa el ELO en valor
  absoluto).
- **`pipeline.py`**: se agregó `EvennessDiscretizer`, que convierte
  `elo_closeness` en una categoría `match_evenness` ∈ {low, medium, high}
  ("low" = partido muy parejo, "high" = diferencia de nivel grande), con
  dos hiperparámetros (`evenness_low_threshold`, `evenness_high_threshold`)
  siguiendo el mismo patrón que `DrawRateDiscretizer`. Se agregó el flag
  `include_evenness` (default `True`) a `create_preprocessing` y
  `create_model_pipeline` para poder comparar el pipeline con y sin el
  atributo nuevo sin duplicar código.

## Metodología del experimento

Para aislar el efecto del atributo nuevo (y no mezclarlo con "se buscó más
combinaciones en el resto del pipeline"), se fijaron **todos los demás
hiperparámetros** en los mejores valores ya encontrados y registrados por
cada notebook:

- **Bayes**: `m=2.0`, `fit_prior=False`, y los márgenes de discretización
  de la búsqueda registrada en `bayesNotebook.ipynb` (F1 macro medio anual
  42.274%).
- **Árbol (ID3)**: `min_info_gain=0.0075` y los márgenes de la búsqueda
  registrada en `experimentos_tree.csv` (F1 macro medio anual 40.183%).

Sobre esa base fija, se corrió una grilla de 10 combinaciones de
`(evenness_low_threshold, evenness_high_threshold)` alrededor de los
percentiles de `elo_closeness` en `train` (mediana ≈ 77, P25 ≈ 34,
P75 ≈ 152), seleccionando la mejor por el mismo criterio que usan los
notebooks (F1 macro medio anual).

## Resultados

### Naive Bayes

| Configuración | F1 macro (media anual) | Accuracy | Balanced accuracy | F1 Empate | Recall Empate |
|---|---|---|---|---|---|
| Baseline (sin `match_evenness`) | 42.27% | 45.70% | 43.28% | 0.25 | 0.21 |
| Con `match_evenness` (low=50, high=150) | 42.30% | 44.69% | 42.84% | 0.27 | 0.25 |

De las 10 combinaciones de umbrales probadas, **solo 1 superó al baseline**,
y por un margen mínimo (+0.0003 en F1 macro). El resto quedó por debajo
(rango 41.7%–42.3%). El recall de empate sí mejora (0.21→0.25), pero a
costa de accuracy y balanced accuracy, que bajan levemente porque el
modelo le quita aciertos a Local y Visitante para dárselos a Empate. En
conjunto: **efecto neto prácticamente nulo para Bayes**, dentro del ruido
esperable entre folds (los propios resultados de la búsqueda de 500
candidatos en el notebook muestran desviaciones estándar anuales de
3-4 puntos porcentuales en F1 macro).

### Árbol de decisión (ID3)

| Configuración | F1 macro (media anual) | Accuracy | Balanced accuracy | F1 Empate | Recall Empate |
|---|---|---|---|---|---|
| Baseline (sin `match_evenness`) | 40.18% | 42.40% | 40.42% | 0.29 | 0.27 |
| Con `match_evenness` (low=20, high=100) | 40.63% | 42.70% | 40.92% | 0.31 | 0.30 |

Acá la mejora es un poco más clara y consistente en dirección (sube F1
macro, accuracy, balanced accuracy y F1/recall de Empate a la vez), pero
igual que con Bayes, **solo 1 de las 10 combinaciones de umbrales superó
al baseline**, y los resultados de la grilla no son monótonos ni estables
respecto al umbral (van de 38.2% a 40.6% de F1 macro sin un patrón suave),
lo que sugiere que el árbol ID3 es sensible a exactamente dónde se corta
`match_evenness` y que parte de esta mejora puede deberse a la variabilidad
de solo 10 años de validación, no a una señal robusta.

**Dato interesante**: al entrenar el árbol completo sobre todo `train`, el
atributo `match_evenness` aparece en el segundo nivel del árbol, en la
rama donde `elo` ya indica ventaja del local (`elo=home`) — ahí el árbol
elige `match_evenness` en vez de `goal_difference` para seguir dividiendo.
Esto indica que el atributo sí aporta información no completamente
redundante con `elo` cuando el local es favorito (para distinguir partidos
"favorito claro" de partidos "favorito, pero parejo"), aunque el impacto
agregado en las métricas sea modesto.

## Conclusión sobre el atributo

El atributo de paridad **no es la mejora que resuelve el problema**, pero
tampoco es negativo: para el árbol da una mejora pequeña y en la dirección
esperada (más recall de empate sin perder mucho en las otras clases); para
Bayes es esencialmente neutro (reparte recall hacia empate a costa de
accuracy). Esto es consistente con la hipótesis original: el problema no es
falta de atributos "de calidad relativa" (de esos ya hay varios, muy
correlacionados entre sí), sino falta de una señal fuerte e independiente
para el empate específicamente, y `abs(elo_difference)` sigue siendo, en el
fondo, una transformación de la misma fuente de información (resultados
históricos) que los demás atributos, no una fuente de información nueva.
Una señal genuinamente nueva (por ejemplo, cuotas de casas de apuestas o
datos a nivel de plantilla) probablemente movería más la aguja, pero no
está disponible en este dataset.

**Recomendación**: vale la pena dejar el atributo disponible (ya que no
perjudica y en el árbol ayuda un poco), pero no esperar que por sí solo
resuelva el techo de rendimiento en la clase Empate. Si se agrega a la
búsqueda de hiperparámetros completa (500 candidatos) de cada notebook,
agregar `preprocessing__evenness__discretizer__low_threshold` y
`preprocessing__evenness__discretizer__high_threshold` (o los nombres
correspondientes de los steps) al `param_grid`.

## Sobre la validación cruzada en años recientes (2013-2022)

Se comparó la distribución de resultados por década en todo el dataset
(1932-2025):

| Década | Empate | Local | Visitante |
|---|---|---|---|
| 1930s | 23.1% | 51.4% | 25.5% |
| 1950s | 24.3% | 51.9% | 23.8% |
| 1970s | 32.6% | 45.4% | 22.0% |
| 1990s | 31.4% | 43.4% | 25.2% |
| 2000s | 25.3% | 39.3% | 35.4% |
| 2010s | 24.2% | 41.5% | 34.3% |
| 2020s | 29.1% | 38.4% | 32.4% |

Y la distribución del propio conjunto de test reservado (2024-2025):
Local 40.2%, Visitante 31.9%, Empate 27.9%.

**La ventaja de localía cayó de forma sostenida**: de ~51-55% de partidos
ganados por el local en 1930-1950 a ~38-41% desde 2000. La proporción de
visitante ganador casi se duplicó (20-25% → 32-35%). Esto no es ruido: es
una tendencia monótona a lo largo de 9 décadas, coherente con lo que se
observa en el fútbol moderno en general (mejores condiciones de viaje,
arbitraje, profesionalización de visitantes, etc.). Como dato adicional,
2020 (temporada sin público por la pandemia) tuvo la tasa de empates más
alta del período reciente (35.4%, contra ~25-28% en los años vecinos) y
una de las tasas de local más bajas — consistente con que la ventaja de
localía depende en parte del público, aunque es un solo año y hay que
tomarlo con cautela por el tamaño de muestra (n=175).

**Esto valida la decisión metodológica de validar (y eventualmente
entrenar) con años recientes en vez de todo el historial**: la
distribución de 2013-2022 (usada para elegir hiperparámetros) se parece
mucho a la del test 2024-2025 reservado, mientras que la distribución de,
por ejemplo, 1950 es claramente distinta. Si hubieran validado con datos
de mediados de siglo XX, estarían optimizando hiperparámetros para un
juego que ya no existe — el "concepto" que se quiere predecir (qué tan
fuerte es la ventaja de local, qué tan parejos son los equipos) cambió con
el tiempo, así que la ventana de entrenamiento/validación recientes es la
elección correcta, no una simplificación.

Dicho esto, hay dos cosas para revisar:

1. **La ventana de entrenamiento de 5 años (`window_years=5`) está fija y
   nunca se probó contra otros valores** (3, 7, 10 años). Se eligió una
   vez y no forma parte de la búsqueda de hiperparámetros, a pesar de que
   probablemente influye tanto o más que los márgenes de discretización
   que sí se buscan exhaustivamente (500 candidatos). Sería razonable
   agregar 2-3 valores de `window_years` como una dimensión más de la
   búsqueda, o al menos correr la búsqueda actual con `window_years=3` y
   `window_years=10` para ver si el resultado es sensible a esa elección.
2. **Solo 10 años de validación (2013-2022) es una muestra chica** para
   elegir entre 500 candidatos con ~10 hiperparámetros: existe riesgo real
   de sobreajustar la elección de hiperparámetros a la idiosincrasia de
   esos 10 años específicos (por algo el propio notebook reporta
   desviaciones estándar de 3-4 puntos entre años, comparables a las
   diferencias entre candidatos). No es un error, pero conviene tenerlo
   presente al interpretar diferencias pequeñas entre configuraciones
   (como las de este mismo experimento) como si fueran mejoras reales.

## Reproducibilidad

Script del experimento (no versionado, corrido puntualmente):
`evenness_experiment.py`. Resultados completos (todas las combinaciones de
umbrales, ambos modelos, reportes por clase) en
`resultados/experimento_evenness_resultados.json`, generado junto a este documento.
No se ejecutó entrenamiento ni predicción sobre el conjunto de test
2024-2025 en ningún paso de este experimento.
