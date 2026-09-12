# Registro de correcciones del dataset

## 2026-09-09 — Fechas de encuentros repetidos

Archivo: `futbol_uruguayo.csv`. Se modificó únicamente `date` en 10 registros. Se conservaron todos los partidos, marcadores, equipos, localías, demás columnas y orden de las filas. Las fuentes respaldan que son encuentros distintos. Esta revisión no certifica el resto del dataset ni verifica si los marcadores incluyen alargue.

### Cambios aplicados

Los nombres y la orientación de equipos son los del CSV. Fechas en formato ISO (AAAA-MM-DD).

| Equipo en home | Equipo en away | Resultado | Fecha anterior | Fecha asignada | Criterio |
|---|---|---|---|---|---|
| Rampla Juniors Futbol Club | Miramar Misiones | 2–1 | 1987-09-26 | 1987-09-27 | Aproximada; primer desempate |
| Miramar Misiones | Rampla Juniors Futbol Club | 1–0 | 1987-09-26 | 1987-09-28 | Aproximada; segundo desempate |
| Rampla Juniors Futbol Club | Miramar Misiones | 1–2 | 1987-09-26 | 1987-09-29 | Aproximada; tercer desempate |
| CA Penarol | Defensor Sporting | 1–1 | 1994-10-01 | 1994-11-06 | Fecha documentada; asignación de fila convencional |
| Defensor Sporting | CA Penarol | 1–1 | 1994-10-01 | 1994-11-13 | Fecha documentada; asignación de fila convencional |
| CA Penarol | Defensor Sporting | 2–1 | 1994-10-01 | 1994-11-20 | Fecha documentada; tercera final |
| Danubio | Nacional | 0–4 | 1996-03-16 | 1996-04-07 | Fecha documentada; Apertura |
| Danubio | Nacional | 1–2 | 1996-03-16 | 1996-11-23 | Fecha documentada; Liguilla |
| CA Penarol | Defensor Sporting | 1–0 | 1997-08-30 | 1997-11-09 | Fecha documentada; primera final |
| CA Penarol | Defensor Sporting | 3–0 | 1997-08-30 | 1997-11-12 | Fecha documentada; segunda final |

### Fuentes y alcance de la evidencia

- **Rampla–Miramar, 1987:** [RSSSF, Uruguay 1987](https://www.rsssf.org/tablesu/uru87.html) registra tres desempates por el descenso: 2–1, 0–1 y 1–2 desde la perspectiva de Rampla. La fuente respalda partidos, resultados y secuencia, pero no sus fechas exactas. **27, 28 y 29 de septiembre son fechas artificiales**, asignadas por solicitud del usuario para separar los encuentros al menos un día y situarlos después del 26/09, que también contiene partidos de la fase regular. No representan una reconstrucción histórica ni un intervalo real de descanso. Quedan pendientes de reemplazo por fechas verificadas.
- **Peñarol–Defensor, 1994:** [1891, Campeonato Uruguayo 1994](https://1891.uy/campeonatos/1/1994/partidos) documenta 1–1 el 06/11, 1–1 el 13/11 y 2–1 el 20/11, todos en cancha neutral. Para los dos empates, las fechas están documentadas, pero no se pudo identificar qué fila original corresponde a cada final: se asignaron siguiendo el orden de aparición en el CSV. Esta correspondencia es convencional, no una localía verificada. No se invirtieron equipos ni se modificaron otras columnas.
- **Danubio–Nacional, 1996:** [LARED21, historial publicado el 14/12/2004](https://www.lr21.com.uy/deportes/162539-ante-nacional-danubio-gana-un-partido-cada-once) registra el 0–4 el 07/04/1996 por el Apertura y el 1–2 el 23/11/1996 por la Liguilla. [RSSSF, Uruguay 1996](https://www.rsssf.org/tablesu/uru96.html) corrobora los resultados y torneos; ubica la jornada de Liguilla el 22–23 de noviembre. Estos dos partidos no son finales de una misma serie.
- **Peñarol–Defensor, 1997:** [1891, Campeonato Uruguayo 1997](https://1891.uy/campeonatos/1/1997/partidos) documenta las finales del 09/11 (1–0) y 12/11 (3–0), en cancha neutral.

### Uso y pendientes

Las fechas aproximadas de 1987 no deben interpretarse como evidencia para calcular días de descanso o establecer una cronología histórica precisa. La asignación individual de los empates de 1994 sigue pendiente de verificación. La condición neutral documentada no se incorporó en esta modificación.

Como se preservó el orden físico del CSV, cualquier cálculo de antecedentes debe ordenar por `date` antes de calcularlos. No se regeneraron archivos derivados, no se ejecutó entrenamiento ni se utilizó el conjunto de test de 2024 en adelante. Los otros conflictos de fechas quedan fuera de esta corrección.

## 2026-09-09 — Revisión de marcadores con P y criterio para E

Criterio solicitado: conservar el marcador al terminar el juego, incluido el alargue si lo hubo, y excluir los goles de la tanda de penales. No equivale necesariamente al resultado a los 90 minutos. Se conservan las marcas `P` y `E`, que describen cómo se definió el encuentro.

### Partidos P contrastados

Los ocho marcadores del CSV ya coinciden con el resultado anterior a la tanda. **No fue necesario modificar `gh` ni `ga` en ningún registro.** Las fechas de la tabla son las del dataset, no una certificación de las fechas locales de disputa.

| Fecha en CSV | home | away | Marcador en CSV y antes de penales | Acción | Fuente |
|---|---|---|---|---|---|
| 2007-05-17 | CA Penarol | Danubio | 1–1 | Conservar | [1891](https://www.1891.uy/partidos/3033/17-05-2007-penarol-1-1-danubio) |
| 2014-06-08 | Montevideo Wanderers | Danubio | 2–2 | Conservar | [RSSSF](https://www.rsssf.org/tablesu/uru2014.html) |
| 2017-12-10 | CA Penarol | Defensor Sporting | 0–0 | Conservar | [AS](https://as.com/futbol/2017/12/10/internacional/1512915019_477900.html) |
| 2017-12-18 | Institucion Atletica Sud America | El Tanque Sisley | 1–1 | Conservar | [Tenfield](https://www.tenfield.com.uy/sud-america-el-tanque-sisley/) |
| 2019-09-08 | Liverpool | River Plate | 2–2 | Conservar | [AUF](https://www.auf.org.uy/liverpool-campeon-del-torneo-intermedio-2019/) |
| 2021-01-15 | Nacional | Montevideo Wanderers | 0–0 | Conservar | [AUF](https://www.auf.org.uy/nacional-y-wanderers-final-intermedio2020-centenario/) |
| 2021-04-01 | Rentistas | Liverpool | 1–1 | Conservar | [AUF](https://www.auf.org.uy/semifinal-campeonato-uruguayo-2020-renliv-estcentenario/) |
| 2021-12-08 | Plaza Colonia | CA Penarol | 1–1 | Conservar | [Tenfield](https://www.tenfield.com.uy/pecu2021cam/) |

En Wanderers–Danubio 2014 y Liverpool–River Plate 2019, el 2–2 corresponde al final del alargue, antes de los penales. Se conserva ese marcador de acuerdo con el criterio solicitado.

### Partidos E conservados por instrucción del usuario

Esta sección registra los valores conservados; no constituye una nueva verificación externa de esos cinco marcadores.

| Fecha en CSV | home | away | Marcador conservado |
|---|---|---|---|
| 2015-06-14 | Nacional | CA Penarol | 3–2 |
| 2016-06-12 | CA Penarol | Plaza Colonia | 3–1 |
| 2018-11-11 | Nacional | CA Penarol | 1–2 |
| 2020-10-15 | Nacional | Rentistas | 0–1 |
| 2022-10-30 | Liverpool | Nacional | 1–4 |

### Observaciones pendientes sobre fechas

Durante la consulta se observaron diferencias de un día entre algunas fechas del CSV y las fuentes: Sud América–El Tanque aparece el 18/12/2017 en CSV y el 17/12 en Tenfield; Nacional–Wanderers aparece el 15/01/2021 en CSV y el 14/01 en AUF; Plaza Colonia–Peñarol aparece el 08/12/2021 en CSV y el 07/12 en Tenfield. No se corrigieron fechas en esta revisión de marcadores. Debe investigarse el criterio horario del dataset antes de normalizarlas.

El CSV se mantuvo idéntico byte por byte en esta revisión. No se modificó `load_base_dataset`, no se ejecutó entrenamiento y no se evaluó el test.
