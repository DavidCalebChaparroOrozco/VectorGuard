from typing import Any, Optional, Union, Callable

# Define the type alias FIRST so Python can use it in the class signature
num = Union[int, float]

class Field:
    def __init__(
        self,
        *,
        gt: Optional[num] = None,                           # Greater than
        ge: Optional[num] = None,                           # Greater than or equal to
        lt: Optional[num] = None,                           # Less than
        le: Optional[num] = None,                           # Less than or equal to
        length: Optional[int] = None,                       # Exact string length
        regex: Optional[str] = None,                        # Regex pattern matching
        nullable: bool = False,                             # Is column allowed to contain nulls?
        repair_with: Optional[Callable[[Any], Any]] = None  # Custom repair logic callback

    ):
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le
        self.length = length
        self.regex = regex
        self.nullable = nullable
        self.repair_with = repair_with