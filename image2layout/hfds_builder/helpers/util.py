import logging
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

logger = logging.getLogger(__name__)


def with_key(key: str, value: Any) -> Any:
    """Associate a key to a collection."""
    return (key, value)


def list_of_dict_to_dict_of_list(values: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert list of dict to dict of list."""
    return {key: [x[key] for x in values] for key in values[0].keys()}


def find_new_split(id_: str, split_ids: dict[str, list[str]]) -> Optional[str]:
    """
    Return a name of a split if the specified id_ belongs to a certain split.
    If id_ does not belong to any split, return None
    """
    for key, values in split_ids.items():
        if id_ in values:
            return key
    return None


def is_area_valid(width: float, height: float, thresh: float = 1e-3) -> bool:
    """
    Filter bounding boxes by its area when it is smaller than the threshold.
    """
    assert 0.0 <= width <= 1.0, width
    assert 0.0 <= height <= 1.0, height
    assert 0.0 <= thresh <= 1.0, thresh
    area = width * height
    valid = area > thresh
    if not valid:
        logger.debug(f"Filtered by area: {area} = width {width} * height {height}")
    return valid


def _compare(low: float, high: float) -> tuple[float, float]:
    if low > high:
        return high, low
    else:
        return low, high


@dataclass
class Coordinates:
    left: float
    center_x: float
    right: float
    width: float
    top: float
    center_y: float
    bottom: float
    height: float

    @staticmethod
    def load_from_cgl_ltwh(
        ltwh: tuple[float, float, float, float],
        global_width: Optional[int] = None,
        global_height: Optional[int] = None,
    ) -> "Coordinates":
        left, top, width, height = ltwh

        if global_width and global_height:
            left /= global_width
            width /= global_width
            top /= global_height
            height /= global_height

        left, right = clamp_w_tol(left), clamp_w_tol(left + width)
        top, bottom = clamp_w_tol(top), clamp_w_tol(top + height)

        left, right = _compare(left, right)
        top, bottom = _compare(top, bottom)

        center_x = clamp_w_tol((left + right) / 2)
        center_y = clamp_w_tol((top + bottom) / 2)
        width, height = right - left, bottom - top

        coordinates = Coordinates(
            left=left,
            center_x=center_x,
            right=right,
            width=width,
            top=top,
            center_y=center_y,
            bottom=bottom,
            height=height,
        )
        coordinates.validate()
        return coordinates

    @staticmethod
    def load_from_pku_ltrb(
        box: tuple[float, float, float, float],
        global_width: Optional[int] = None,
        global_height: Optional[int] = None,
    ) -> "Coordinates":
        left, top, right, bottom = box

        if global_width and global_height:
            left /= global_width
            right /= global_width
            top /= global_height
            bottom /= global_height

        left, right = clamp_w_tol(left), clamp_w_tol(right)
        top, bottom = clamp_w_tol(top), clamp_w_tol(bottom)

        left, right = _compare(left, right)
        top, bottom = _compare(top, bottom)

        center_x = clamp_w_tol((left + right) / 2)
        center_y = clamp_w_tol((top + bottom) / 2)
        width, height = right - left, bottom - top

        coordinates = Coordinates(
            left=left,
            center_x=center_x,
            right=right,
            width=width,
            top=top,
            center_y=center_y,
            bottom=bottom,
            height=height,
        )
        coordinates.validate()
        return coordinates

    def validate(self) -> None:
        """
        Check if all values are valid (i.e. in [0, 1] range).
        """
        assert 0.0 <= self.left <= 1.0, self
        assert 0.0 <= self.center_x <= 1.0, self
        assert 0.0 <= self.right <= 1.0, self
        assert 0.0 <= self.width <= 1.0, self
        assert self.left <= self.right, self

        assert 0.0 <= self.top <= 1.0, self
        assert 0.0 <= self.center_y <= 1.0, self
        assert 0.0 <= self.bottom <= 1.0, self
        assert 0.0 <= self.height <= 1.0, self
        assert self.top <= self.bottom, self

    def has_valid_area(self, thresh: float = 1e-3) -> bool:
        """
        Check whether the area is smaller than the threshold.
        """
        area = self.width * self.height
        valid = area > thresh
        if not valid:
            logger.debug(f"Filtered by {area=} = {self.width=} * {self.height=}")
        return valid


def clamp_w_tol(
    value: float, tolerance: float = 5e-3, vmin: float = 0.0, vmax: float = 1.0
) -> float:
    """
    Clamp the value to [vmin, vmax] range with tolerance.
    """
    assert vmin - tolerance <= value <= vmax + tolerance, value
    return max(vmin, min(vmax, value))


class Element(NamedTuple):
    label: str
    coordinates: Coordinates


class Sample(NamedTuple):
    id: str
    identifier: str
    image_width: int
    image_height: int
    elements: list[Element] = []
    split: Optional[str] = None
