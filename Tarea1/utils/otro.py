import pandas as pd

from decisionTree import getTree, predict


# Ejemplo clasico de ID3: decidir si se juega al aire libre.
training_table = pd.DataFrame(
    [
        ["Soleado", "Calor", "Alta", "Debil", "No"],
        ["Soleado", "Calor", "Alta", "Fuerte", "No"],
        ["Nublado", "Calor", "Alta", "Debil", "Si"],
        ["Lluvioso", "Templado", "Alta", "Debil", "Si"],
        ["Lluvioso", "Frio", "Normal", "Debil", "Si"],
        ["Lluvioso", "Frio", "Normal", "Fuerte", "No"],
        ["Nublado", "Frio", "Normal", "Fuerte", "Si"],
        ["Soleado", "Templado", "Alta", "Debil", "No"],
        ["Soleado", "Frio", "Normal", "Debil", "Si"],
        ["Lluvioso", "Templado", "Normal", "Debil", "Si"],
        ["Soleado", "Templado", "Normal", "Fuerte", "Si"],
        ["Nublado", "Templado", "Alta", "Fuerte", "Si"],
        ["Nublado", "Calor", "Normal", "Debil", "Si"],
        ["Lluvioso", "Templado", "Alta", "Fuerte", "No"],
    ],
    columns=["Pronostico", "Temperatura", "Humedad", "Viento", "result"],
)

attributes = ["Pronostico", "Temperatura", "Humedad", "Viento"]

decision_tree = getTree(attributes, training_table, min_info_gain=0)
decision_tree.print_tree()


# Ejemplos nuevos para clasificar con el arbol ya entrenado.
prediction_examples = [
    {
        "Pronostico": "Soleado",
        "Temperatura": "Calor",
        "Humedad": "Alta",
        "Viento": "Debil",
    },
    {
        "Pronostico": "Soleado",
        "Temperatura": "Frio",
        "Humedad": "Normal",
        "Viento": "Fuerte",
    },
    {
        "Pronostico": "Nublado",
        "Temperatura": "Templado",
        "Humedad": "Alta",
        "Viento": "Fuerte",
    },
    {
        "Pronostico": "Lluvioso",
        "Temperatura": "Frio",
        "Humedad": "Alta",
        "Viento": "Debil",
    },
    {
        "Pronostico": "Lluvioso",
        "Temperatura": "Calor",
        "Humedad": "Normal",
        "Viento": "Fuerte",
    },
]

expected_results = ["No", "Si", "Si", "Si", "No"]

print("\nPredicciones:")
for number, (row, expected) in enumerate(
    zip(prediction_examples, expected_results), start=1
):
    predicted = predict(decision_tree, row)
    print(
        f"Ejemplo {number}: prediccion={predicted}, "
        f"esperado={expected}, correcto={predicted == expected}"
    )
