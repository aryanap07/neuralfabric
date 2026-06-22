from neuralfabric.core.tensor import Tensor
from neuralfabric.linear_model import LogisticRegression
from neuralfabric.model_selection import train_test_split


def main() -> None:
    """
    Customer Purchase Prediction

    Features:
        - Age
        - Annual Income ($1000s)

    Target:
        - 1 = Purchased
        - 0 = Not Purchased
    """

    X = Tensor(
        [
            [18, 20],
            [22, 25],
            [25, 30],
            [30, 35],
            [35, 45],
            [40, 55],
            [45, 65],
            [50, 75],
        ]
    )

    y = Tensor(
        [
            [0],
            [0],
            [0],
            [0],
            [1],
            [1],
            [1],
            [1],
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = LogisticRegression(
        lr=0.001,
        epochs=5000,
    )

    model.fit(X_train, y_train)

    customer = Tensor(
        [
            [38, 50],
        ]
    )

    probability = model.predict_proba(customer)
    prediction = model.predict(customer)

    print("Purchase Probability:")
    print(probability)

    print()

    print("Prediction:")
    print(prediction)

    print()

    print("Test Accuracy:")
    print(model.score(X_test, y_test))


if __name__ == "__main__":
    main()