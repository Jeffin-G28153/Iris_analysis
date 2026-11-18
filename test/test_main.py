from main import load_iris_data


def test_dataset_has_150_rows():
    df = load_iris_data()

    assert len(df) == 150


def test_dataset_has_five_columns():
    df = load_iris_data()

    assert len(df.columns) == 5


def test_species_column_exists():
    df = load_iris_data()

    assert "species" in df.columns


def test_dataset_has_three_species():
    df = load_iris_data()

    assert df["species"].nunique() == 3


def test_dataset_has_no_missing_values():
    df = load_iris_data()

    assert df.isnull().sum().sum() == 0