from sklearn.datasets import load_iris
import pandas as pd


def load_iris_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = [
        iris.target_names[i]
        for i in iris.target
    ]

    return df


def dataset_summary(df):
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSpecies Count:")
    print(df["species"].value_counts())


def average_sepal_length(df):
    print(
        "\nAverage Sepal Length:",
        df["sepal length (cm)"].mean()
    )

def find_maximum_petal_length(df):
    print(
        "\nMaximum Petal Length:",
        df["petal length (cm)"].max()
    )


def main():
    df = load_iris_data()
    dataset_summary(df)
    average_sepal_length(df)
    find_maximum_petal_length(df)

if __name__ == "__main__":
    main()