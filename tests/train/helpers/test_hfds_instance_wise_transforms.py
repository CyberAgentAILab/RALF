from image2layout.train.helpers.hfds_instance_wise_transforms import (
    get_indexes_for_lexicographic_sort,
    sort_label_transform,
)


def test_sort_label_transform() -> None:
    inputs_w_1ddata = {
        "label": [1, 2, 0],
        "data": ["a", "b", "c"],
    }
    expected_w_1ddata = {
        "label": [0, 1, 2],
        "data": ["c", "a", "b"],
    }

    inputs_w_2ddata = {
        "label": [1, 0],
        "data": [[0, 1], [2, 3]],
    }
    expected_w_2ddata = {
        "label": [0, 1],
        "data": [[2, 3], [0, 1]],
    }
    assert sort_label_transform(inputs_w_1ddata) == expected_w_1ddata
    assert sort_label_transform(inputs_w_2ddata) == expected_w_2ddata


def test_get_indexes_lexicographic_sort() -> None:
    inputs = {
        "center_x": [0.5, 0.5, 0.4],
        "center_y": [0.5, 0.3, 0.3],
        "width": [1.0, 0.5, 0.5],
        "height": [0.8, 0.6, 0.6],
    }
    expected = [2, 1, 0]
    assert get_indexes_for_lexicographic_sort(inputs) == expected


if __name__ == "__main__":
    test_sort_label_transform()
    test_get_indexes_lexicographic_sort()
