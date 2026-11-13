from typing import Any, Callable, Optional, Union

# Define the type alias FIRST so Python can use it in the class signature
num = Union[int, float]


class Field:
    """Declarative field specification for DataContract schemas."""

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
        self.data_type: Optional[type] = None

    def clone_with_type(self, data_type: type, force_nullable: bool = False) -> "Field":
        """
        Creates an independent copy of this Field bound to a resolved data_type.
        Prevents shared Field instances from mutating each other across contracts.
        """
        cloned = Field(
            gt=self.gt,
            ge=self.ge,
            lt=self.lt,
            le=self.le,
            length=self.length,
            regex=self.regex,
            nullable=True if force_nullable else self.nullable,
            repair_with=self.repair_with,
        )
        cloned.data_type = data_type
        return cloned
