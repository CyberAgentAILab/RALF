from image2layout.train.helpers.util import argsort


def test_argsort() -> None:
    input_, expected = [0.3, 1.0, 0.5], [0, 2, 1]
    assert argsort(input_) == expected


if __name__ == "__main__":
    test_argsort()
