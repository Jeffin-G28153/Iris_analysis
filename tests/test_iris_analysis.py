import pandas as pd

from main import (
    average_petal_length_by_species,
    average_sepal_length,
    dataset_summary,
    find_maximum_petal_length,
    load_iris_data,
)


def test_load_iris_data():
    df = load_iris_data()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (150, 5)

    assert df.columns.tolist() == [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
    ]

    assert df["species"].value_counts().to_dict() == {
        "setosa": 50,
        "versicolor": 50,
        "virginica": 50,
    }

    assert not df.isna().any().any()


def test_dataset_summary(capsys):
    df = pd.DataFrame({
        "sepal length (cm)": [4.0, 6.0, 8.0],
        "species": ["setosa", "setosa", "virginica"],
    })

    dataset_summary(df)

    output = capsys.readouterr().out

    assert "Shape: (3, 2)" in output
    assert "['sepal length (cm)', 'species']" in output
    assert "Species Count:" in output

    # Check the reported counts without depending on spacing.
    lines = [line.split() for line in output.splitlines()]
    assert ["setosa", "2"] in lines
    assert ["virginica", "1"] in lines


def test_average_sepal_length(capsys):
    df = pd.DataFrame({
        "sepal length (cm)": [4.0, 6.0, 8.0],
    })

    average_sepal_length(df)

    output = capsys.readouterr().out

    assert output.strip() == "Average Sepal Length: 6.0"


def test_find_maximum_petal_length(capsys):
    df = pd.DataFrame({
        "petal length (cm)": [1.0, 5.0, 3.0],
    })

    find_maximum_petal_length(df)

    output = capsys.readouterr().out

    assert output.strip() == "Maximum Petal Length: 5.0"


def test_average_petal_length_by_species(capsys):
    df = pd.DataFrame({
        "species": [
            "setosa",
            "setosa",
            "virginica",
            "virginica",
        ],
        "petal length (cm)": [1.0, 3.0, 5.0, 7.0],
    })

    average_petal_length_by_species(df)

    output = capsys.readouterr().out

    assert "Average Petal Length by Species:" in output

    lines = [line.split() for line in output.splitlines()]
    assert ["setosa", "2.0"] in lines
    assert ["virginica", "6.0"] in lines