# Atributo de paridad

Se define `elo_closeness = abs(elo_difference)` y se discretiza como `match_evenness`. Se busca representar cuán parejos son los equipos, independientemente de cuál tiene mayor rating. Es una transformación de información existente, no una fuente adicional.

## Experimento histórico: 2013–2022

Se fijaron los otros hiperparámetros y se evaluaron diez pares de umbrales. No se utilizaron métricas de test para calcular esta comparación.

| Modelo | F1 macro anual sin paridad | Mejor F1 con paridad | Recall empate sin/con |
| --- | ---: | ---: | ---: |
| Bayes | 42,27% | 42,30% | 21% / 25% |
| ID3 | 40,18% | 40,63% | 27% / 30% |

Solo una de las diez combinaciones superó la referencia en cada modelo. La ganancia se interpreta como pequeña y sensible a los umbrales, sin afirmar una mejora robusta. El atributo se mantiene disponible y sus umbrales se seleccionan dentro del pipeline.

El script actual incluye 2005–2023. Los valores anteriores corresponden al protocolo histórico, conservado en `resultados/experimento_evenness_resultados.json`; una ejecución nueva genera un archivo separado.
