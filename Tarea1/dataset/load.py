import pandas as pd


#funcion que se usa para generar nuevos atributos a partir del dataset origial
def load_attributes(
    dataset: pd.DataFrame,
    years_limit: int = 1,
    matches_limit: int = 5
) -> pd.DataFrame:

    dataset = dataset.copy()

    #funcion que obtiene el historial de partidos de un equipo hasta una fecha determinada
    def get_record(
        dataset: pd.DataFrame,
        years_limit: int,
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

    #funcion que obtiene el historial directo entre dos equipos
    def get_h2h_record(
        dataset: pd.DataFrame,
        years_limit: int,
        date: pd.Timestamp,
        team1: str,
        team2: str
    ) -> pd.DataFrame:

        start_date = date - pd.DateOffset(
            years=years_limit
        )

        h2h = dataset[
            (
                ((dataset["home"] == team1) & (dataset["away"] == team2))
                | ((dataset["home"] == team2) & (dataset["away"] == team1))
            )
            & (dataset["date"] >= start_date)
            & (dataset["date"] < date)
        ]

        return h2h.sort_values("date")

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

    #funcion que obtiene la tasa de empates de un historial de partidos
    def get_draw_rate(
        record: pd.DataFrame
    ) -> float:

        if len(record) == 0:
            return 0.0

        draws = (
            record["result"] == "E"
        ).sum()

        return draws / len(record)

    #funcion que obtiene los goles a favor de un equipo cuando juega como local
    def goals_for_local(
        record: pd.DataFrame,
        team: str
    ) -> float:
        local_matches = record[
            record["home"] == team
        ]

        if len(local_matches) == 0:
            return 0.0

        return float(
            local_matches["gh"].mean()
        )

    #funcion que obtiene los goles recibidos por un equipo cuando juega como local
    def goals_against_local(
        record: pd.DataFrame,
        team: str
    ) -> float:
        local_matches = record[
            record["home"] == team
        ]

        if len(local_matches) == 0:
            return 0.0

        return float(
            local_matches["ga"].mean()
        )

    #funcion que obtiene los goles a favor de un equipo cuando juega como visitante
    def goals_for_away(
        record: pd.DataFrame,
        team: str
    ) -> float:
        away_matches = record[
            record["away"] == team
        ]

        if len(away_matches) == 0:
            return 0.0

        return float(
            away_matches["ga"].mean()
        )

    #funcion que obtiene los goles recibidos por un equipo cuando juega como visitante
    def goals_against_away(
        record: pd.DataFrame,
        team: str
    ) -> float:
        away_matches = record[
            record["away"] == team
        ]

        if len(away_matches) == 0:
            return 0.0

        return float(
            away_matches["gh"].mean()
        )

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
            dataset,
            years_limit,
            date,
            home_team
        )
        away_record = get_record(
            dataset,
            years_limit,
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

        #positivo significa mejor historial del local y negativo mejor
        #historial del visitante
        record_difference = home_rate - away_rate

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

        #positivo significa mejor forma reciente del local y negativo mejor
        #forma reciente del visitante
        last_matches_difference = home_condition - away_condition

        home_goals_for = goals_for_local(
            home_record,
            home_team
        )

        home_goals_against = goals_against_local(
            home_record,
            home_team
        )

        away_goals_for = goals_for_away(
            away_record,
            away_team
        )

        away_goals_against = goals_against_away(
            away_record,
            away_team
        )

        home_goal_difference = home_goals_for - home_goals_against
        away_goal_difference = away_goals_for - away_goals_against

        #comparamos al local jugando como local contra el visitante jugando
        #como visitante
        goal_difference_value = (
            home_goal_difference - away_goal_difference
        )

        #positivo significa que el local convierte mas goles en casa que el
        #visitante jugando fuera
        attack_difference = home_goals_for - away_goals_for

        #se resta en este orden porque recibir menos goles significa defender
        #mejor; positivo representa una mejor defensa del local
        defense_difference = away_goals_against - home_goals_against

        home_draw_rate = get_draw_rate(
            home_record
        )

        away_draw_rate = get_draw_rate(
            away_record
        )

        #este atributo no compara equipos: mide la tendencia conjunta al empate
        draw_rate_average = (
            home_draw_rate + away_draw_rate
        ) / 2

        #historial directo (5 años)
        h2h_record = get_h2h_record(
            dataset,
            5,
            date,
            home_team,
            away_team
        )
        home_h2h_points = get_points_rate(h2h_record, home_team)
        away_h2h_points = get_points_rate(h2h_record, away_team)
        h2h_difference = home_h2h_points - away_h2h_points

        home_rest_days = (
            (date - home_record.iloc[-1]["date"]).days
            if len(home_record) > 0
            else 30
        )
        away_rest_days = (
            (date - away_record.iloc[-1]["date"]).days
            if len(away_record) > 0
            else 30
        )
        
        #positivo significa que el local tuvo mas dias de descanso (ventaja local)
        rest_days_difference = home_rest_days - away_rest_days

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
            "record_difference": record_difference,
            "last_matches_difference": last_matches_difference,
            "goal_difference_value": goal_difference_value,
            "attack_difference": attack_difference,
            "defense_difference": defense_difference,
            "draw_rate_average": draw_rate_average,
            "h2h_difference": h2h_difference,
            "rest_days_difference": rest_days_difference,
            "local_experience": local_experience,
            "away_experience": away_experience,
            "record_enough": record_enough,
        })

    new_attributes = dataset.apply(
        calculate_attributes,
        axis=1
    )

    # Calculate ELO sequentially
    elo_dict = {}
    elo_differences = []
    
    K = 20
    
    for idx, row in dataset.iterrows():
        home = row['home']
        away = row['away']
        
        home_elo = elo_dict.get(home, 1500.0)
        away_elo = elo_dict.get(away, 1500.0)
        
        # Guardamos la diferencia PREVIA al partido
        elo_diff = home_elo - away_elo
        elo_differences.append(elo_diff)
        
        # Actualizamos para la proxima fecha
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home
        
        if row['gh'] > row['ga']:
            actual_home, actual_away = 1.0, 0.0
        elif row['gh'] < row['ga']:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
            
        home_elo_new = home_elo + K * (actual_home - expected_home)
        away_elo_new = away_elo + K * (actual_away - expected_away)
        
        elo_dict[home] = home_elo_new
        elo_dict[away] = away_elo_new

    new_attributes["elo_difference"] = elo_differences

    #atributo de paridad: que tan pareja es la diferencia de nivel entre los
    #equipos, sin importar quien es mejor. Se calcula en base al valor
    #absoluto de la diferencia de ELO, que es el resumen mas directo del
    #nivel de cada equipo. A diferencia de los atributos anteriores (que
    #indican quien es mejor), este busca capturar directamente que tan
    #cerrado esta el partido, senal que hoy no esta representada de forma
    #explicita y que puede ayudar a distinguir empates.
    new_attributes["elo_closeness"] = [abs(value) for value in elo_differences]

    return pd.concat(
        [dataset, new_attributes],
        axis=1
    )

def load_dataset(name_dataset, years_limit=1, matches_limit=5):
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

    df = load_attributes(df, years_limit=years_limit, matches_limit=matches_limit)

    return df

def load_base_dataset(name_dataset):
    #leemos el dataset de futbol uruguayo
    df = pd.read_csv(name_dataset)

    #nos quedamos con las columnas que nos interesan para el clasificador
    df = df.drop(columns=["full_time", "competition", "home_ident", "away_ident", "home_country", "away_country", "home_code", "away_code", "home_continent", "away_continent", "continent", "level"])

    #convertimos las columnas a los tipos de datos correctos
    df["date"] = pd.to_datetime(df["date"], errors="raise")

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

    return df
