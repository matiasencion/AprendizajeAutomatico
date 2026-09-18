# Para el informe: por qué se extendió la validación cruzada a 18 años

*(Texto listo para copiar/adaptar en el informe. Sección sugerida: "Selección
de hiperparámetros" o "Validación cruzada".)*

## El problema: elegir entre muchos candidatos con pocos años de validación

La búsqueda de hiperparámetros evalúa cada configuración candidata
entrenando y validando en varios *folds* temporales (un fold por año:
se entrena con los años anteriores y se valida en ese año). El resultado
de cada candidato no es un solo número, sino un conjunto de valores de
F1 macro, uno por año de validación. Reportar sólo el promedio esconde
qué tan **variable** es ese promedio de un año a otro, y esa variabilidad
es justamente lo que determina si diferencias pequeñas entre candidatos
son reales o son ruido.

## Desviación estándar y error estándar

Sea $s_1, s_2, \dots, s_n$ el F1 macro obtenido en cada uno de los $n$ años
de validación para una configuración de hiperparámetros. Se calculan:

**Media entre años:**

$$\bar{s} = \frac{1}{n} \sum_{i=1}^{n} s_i$$

**Desviación estándar muestral** (qué tan dispersos están los años entre sí):

$$\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (s_i - \bar{s})^2}$$

**Error estándar de la media** (qué tan preciso es el promedio $\bar{s}$
como estimación del desempeño "verdadero" de esa configuración):

$$SE = \frac{\sigma}{\sqrt{n}}$$

El error estándar es lo relevante para *comparar candidatos*: dos
configuraciones cuyos promedios difieren en menos de, aproximadamente,
un $SE$ no se pueden distinguir con confianza a partir de esos datos —
la diferencia observada puede deberse simplemente a en qué años les tocó
validar.

## Por qué importa: la fórmula muestra que $n$ chico infla la incertidumbre

Como $SE$ depende de $1/\sqrt{n}$, con pocos años de validación el error
estándar es grande aunque la desviación estándar entre años ($\sigma$) sea
moderada. Con la validación original (10 años, 2013-2022) se observaron
desviaciones estándar de F1 macro de aproximadamente 3-4 puntos
porcentuales entre años — una magnitud comparable a la diferencia entre el
candidato #1 y el candidato #10 del ranking de 500 candidatos probados.
En otras palabras: con sólo 10 años, **el "ganador" de la búsqueda de
hiperparámetros no se podía distinguir con confianza de varios de los
candidatos siguientes**; parte de esa elección podía deberse a qué años
específicos se usaron para validar, no a que esa configuración fuera
realmente mejor.

## La corrección aplicada

Se extendió el rango de años de validación de **2013-2022 (10 años)** a
**2005-2022 (18 años)**. Esto no es arbitrario: se verificó primero que la
distribución de resultados (Local/Empate/Visitante) de la década de 2000
ya es similar a la actual, y muy distinta a la de décadas más antiguas
(donde la ventaja de localía era mucho mayor) — de modo que agregar esos
años no mezcla un "fútbol" distinto con el actual.

Al pasar de $n=10$ a $n=18$, el error estándar se reduce en un factor de
$\sqrt{10/18} \approx 0{,}75$ sólo por el efecto del tamaño de muestra —
la desviación estándar entre años ($\sigma$) no baja por agregar folds (es
una propiedad del dominio, cuánto varía el fútbol de un año a otro), pero
el error estándar de la media sí, porque se está promediando sobre más
años. Se volvió a correr la búsqueda completa de 500 candidatos con las 18
divisiones y se comparó el candidato ganador de cada modelo antes y
después:

| Modelo | $\sigma$ (10 años) | $SE$ (10 años) | $\sigma$ (18 años) | $SE$ (18 años) | Reducción de $SE$ |
|---|---|---|---|---|---|
| Naive Bayes | 0,0366 | 0,0116 | 0,0326 | 0,0077 | ≈34% |
| Árbol (ID3) | 0,0324 | 0,0102 | 0,0299 | 0,0071 | ≈31% |

En los dos modelos la reducción real (31-34%) es un poco mayor a la que
predice sólo el tamaño de muestra (25%, por $\sqrt{10/18}$), lo que indica
que el conjunto 2005-2022 en su conjunto también es levemente menos
errático año a año que si se hubiese usado, por ejemplo, un tramo con años
atípicos (como 2020, con la pandemia).

Además, con F1 macro medio anual prácticamente igual antes y después
(Bayes: 42,27% → 42,70%; Árbol: 40,18% → 40,04%), la extensión de años
**no cambió la conclusión sustantiva** — es una mejora en qué tan
confiable es la estimación, no un cambio en qué tan bueno es el modelo.

## Criterio de selección más robusto: la regla de "1 error estándar"

Además de extender los años, se adoptó un criterio de selección más
conservador que el simple máximo. En vez de quedarse directamente con el
candidato de mayor F1 macro promedio, se calcula el error estándar del
mejor candidato y se consideran **estadísticamente equivalentes** a todos
los candidatos cuyo promedio cae dentro de un $SE$ del mejor:

$$\text{equivalentes} = \{\, \text{candidato } j \;:\; \bar{s}_j \geq \bar{s}_{\text{mejor}} - SE_{\text{mejor}} \,\}$$

Entre ese grupo de candidatos empatados, se elige el que responda a un
criterio adicional razonado (por ejemplo, el hiperparámetro que representa
un modelo más simple o más conservador), en vez de asumir que el máximo
puntual es necesariamente la mejor configuración. Esta regla es estándar
en selección de modelos (se usa, por ejemplo, en `glmnet` para elegir la
regularización) y evita sacar conclusiones de diferencias que están
dentro del margen de ruido de la propia validación cruzada.

**Aplicado a los resultados reales:** con $SE=0{,}0077$ (Bayes, 18 años),
el umbral de 1-SE es $0{,}42695 - 0{,}0077 = 0{,}4192$; los primeros **10
candidatos del ranking** (de 500) tienen F1 macro entre 0,4230 y 0,4270,
todos por encima de ese umbral. Con el árbol pasa lo mismo: umbral
$0{,}40037-0{,}0071=0{,}3933$, y los 10 primeros candidatos (0,3954 a
0,4004) también caen todos dentro. **En ningún caso el ganador
"seleccionado" es distinguible del resto del top-10** — es un hallazgo
concreto, no hipotético, de que declarar un único ganador por el máximo
puntual transmite más precisión de la que realmente hay.

## Resumen para citar

> Se extendió la validación cruzada temporal de 10 a 18 años (2005-2022)
> para reducir el error estándar de la estimación de F1 macro de cada
> candidato de hiperparámetros, dado que con sólo 10 años esa incertidumbre
> ($SE\approx0{,}010$-$0{,}012$ según el modelo) era comparable a la
> diferencia entre los primeros candidatos del ranking de 500 probados. Al
> repetir la búsqueda completa con 18 años, el error estándar del ganador
> bajó a $SE\approx0{,}007$-$0{,}008$ en ambos modelos (reducción de
> 31-34%, consistente con el factor $\sqrt{10/18}\approx0{,}75$ esperado
> por la fórmula $SE=\sigma/\sqrt{n}$), sin cambiar la conclusión
> sustantiva: el F1 macro medio anual del ganador prácticamente no varió
> (Bayes 42,27%→42,70%; árbol 40,18%→40,04%). Se verificó previamente que
> la distribución de resultados de 2005 en adelante es consistente con la
> actual, a diferencia de décadas anteriores. Adicionalmente, se aplicó un
> criterio de selección basado en "1 error estándar", que muestra que en
> los dos modelos los primeros 10 candidatos del ranking de 500 son
> estadísticamente indistinguibles entre sí.
