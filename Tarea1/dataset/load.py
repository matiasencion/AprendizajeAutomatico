import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np


#funcion que se usa para generar nuevos atributos a partir del dataset origial
def load_attributes(
    dataset: pd.DataFrame,
    years_limit: int = 1,
    matches_limit: int = 5
) -> pd.DataFrame:

    dataset = dataset.copy()

    #funcion que obtiene el historial de partidos de un equipo hasta una fecha determinada
    def get_record(
        date: pd.Timestamp,
        team: str
    ) -> pd.DataFrame:

        start_date = date - pd.DateOffset(
            years=years_limit
        )

        record = dataset[
            (
                (dataset["home"] == team)
                | (dataset["away"] == team)
            )
            & (dataset["date"] >= start_date)
            & (dataset["date"] < date)
        ]

        return record.sort_values("date")

    #funcion que obtiene la cantidad de victorias de un equipo en un historial de partidos
    def get_wins(
        record: pd.DataFrame,
        team: str
    ) -> int:

        home_victories = (
            (record["home"] == team)
            & (record["result"] == "L")
        ).sum()

        away_victories = (
            (record["away"] == team)
            & (record["result"] == "V")
        ).sum()

        return int(
            home_victories + away_victories
        )

    #funcion que obtiene la tasa de victorias de un equipo en un historial de partidos
    def get_win_rate(
        record: pd.DataFrame,
        team: str
    ) -> float:

        if len(record) == 0:
            return 0.0

        return get_wins(record, team) / len(record)

    #funcion que obtiene la tasa de puntos de un equipo en un historial de partidos
    def get_points_rate(
        record: pd.DataFrame,
        team: str
    ) -> float:

        if len(record) == 0:
            return 0.0

        home_points = (
            (record["home"] == team)
            & (record["result"] == "L")
        ).sum() * 3 + (
            (record["home"] == team)
            & (record["result"] == "E")
        ).sum()

        away_points = (
            (record["away"] == team)
            & (record["result"] == "V")
        ).sum() * 3 + (
            (record["away"] == team)
            & (record["result"] == "E")
        ).sum()

        return (home_points + away_points) / (len(record) * 3)

    #funcion que obtiene la diferencia de goles de un equipo en un historial de partidos
    def goal_difference_local(
        record: pd.DataFrame,
        team: str
    ) -> float:
        local_matches = record[
            record["home"] == team
        ]

        if len(local_matches) == 0:
            return 0.0

        goals_for = local_matches["gh"].mean()
        goals_against = local_matches["ga"].mean()

        return float(goals_for - goals_against)

    #funcion que obtiene la diferencia de goles de un equipo en un historial de partidos
    def goal_difference_away(
        record: pd.DataFrame,
        team: str
    ) -> float:
        away_matches = record[
            record["away"] == team
        ]

        if len(away_matches) == 0:
            return 0.0

        goals_for = away_matches["ga"].mean()
        goals_against = away_matches["gh"].mean()

        return float(goals_for - goals_against)

    #funcion que compara dos tasas y devuelve "L" si la primera es mayor, "V" si la segunda es mayor y "E" si son iguales
    def compare(
        home_rate: float,
        away_rate: float
    ) -> str:

        if home_rate > away_rate:
            return "L"
        elif home_rate < away_rate:
            return "V"
        return "E"

    #funcion que devuelve un nivel de experiencia en base a la cantidad de partidos jugados
    def level_experience(quantity: int) -> int:
        if quantity == 0:
            return 0  # sin información
        elif quantity < 3:
            return 1  # poca información
        elif quantity < 5:
            return 2  # información intermedia
        else:
            return 3  # información suficiente

    #funcion que calcula los nuevos atributos para un partido dado
    def calculate_attributes(
        row: pd.Series
    ) -> pd.Series:

        date = row["date"]
        home_team = row["home"]
        away_team = row["away"]

        home_record = get_record(
            date,
            home_team
        )
        away_record = get_record(
            date,
            away_team
        )

        home_rate = get_win_rate(
            home_record,
            home_team
        )
        away_rate = get_win_rate(
            away_record,
            away_team
        )

        ventaja_historica = compare(
            home_rate,
            away_rate
        )

        home_lasts = home_record.tail(
            matches_limit
        )
        away_lasts = away_record.tail(
            matches_limit
        )

        home_condition = get_points_rate(
            home_lasts,
            home_team
        )
        away_condition = get_points_rate(
            away_lasts,
            away_team
        )

        last_matches = compare(
            home_condition,
            away_condition
        )

        goal_difference = compare(
            goal_difference_local(home_record, home_team),
            goal_difference_away(away_record, away_team)
        )

        local_experience = level_experience(
            len(home_record)
        )

        away_experience = level_experience(
            len(away_record)
        )

        record_enough = int(
            len(home_record) >= matches_limit
            and len(away_record) >= matches_limit
        )

        return pd.Series({
            "record": ventaja_historica,
            "last_matches": last_matches,
            "goal_difference": goal_difference,
            "local_experience": local_experience,
            "away_experience": away_experience,
            "record_enough": record_enough,
        })

    new_attributes = dataset.apply(
        calculate_attributes,
        axis=1
    )

    return pd.concat(
        [dataset, new_attributes],
        axis=1
    )

def load_dataset(name_dataset):
    #leemos el dataset de futbol uruguayo
    df = pd.read_csv(name_dataset)

    #nos quedamos con las columnas que nos interesan para el clasificador
    df = df.drop(columns=["full_time", "competition", "home_ident", "away_ident", "home_country", "away_country", "home_code", "away_code", "home_continent", "away_continent", "continent", "level"])

    #convertimos las columnas a los tipos de datos correctos
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["gh"] = pd.to_numeric(df["gh"], errors="raise").astype(int)
    df["ga"] = pd.to_numeric(df["ga"], errors="raise").astype(int)

    #ordenamos el dataset por fecha y por equipos, y eliminamos duplicados
    df = (
        df.drop_duplicates()
        .sort_values(["date", "home", "away"], kind="stable")
        .reset_index(drop=True)
    )

    #creamos la columna result, que es el resultado del partido, L si gana el local, V si gana el visitante y E si empatan
    df["result"] = df.apply(
        lambda row: "L" if row["gh"] > row["ga"] else ("V" if row["gh"] < row["ga"] else "E"), axis=1
    )

    df = load_attributes(df)

    return df