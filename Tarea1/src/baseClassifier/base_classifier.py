import pandas as pd

#porcentaje de victorias de un equipo en un historial de partidos (como local o visitante)
def win_rate(record: pd.DataFrame, team: str) -> float:
    if len(record) == 0:
        return 0.0

    count = 0
    for match in record.itertuples():
        if match.home == team and match.gh > match.ga:
            count += 1
        elif match.away == team and match.ga > match.gh:
            count += 1

    return count / len(record)

#funcion que obtiene el historial de partidos de un equipo hasta una fecha determinada
def get_record(
    dataset: pd.DataFrame,
    years_limit: int,
    date: pd.Timestamp,
    team: str
) -> pd.DataFrame:
    
    dataset = dataset.copy()

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

def base_classifier(
    dataset: pd.DataFrame,
    years_limit: int,
    row: pd.Series,
) -> str:

    home_record = get_record(dataset, years_limit, row.date, row.home)
    away_record = get_record(dataset, years_limit, row.date, row.away)

    home_win_rate = win_rate(home_record, row.home)
    away_win_rate = win_rate(away_record, row.away)

    if home_win_rate > away_win_rate:
        return "L"
    elif home_win_rate < away_win_rate:
        return "V"
    else:
        return "E"