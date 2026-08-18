from typing import Any, Optional

class Field:
    def __init__(
        self,
        *,
        gt: Optional[num] = None,       # Greater than
        ge: Optional[num] = None,       # Greater than or equal to
        lt: Optional[num] = None,       # Less than
        le: Optional[num] = None,       # Less than or equal to
        length: Optional[int] = None,   # Exact string length
        regex: Optional[str] = None,    # Regex pattern matching
        nullable: bool = False          # Is column allowed to contain nulls?
    ):
        self.gt = gt
        self.ge = ge
        self.lt = lt
        self.le = le
        self.length = length
        self.regex = regex
        self.nullable = nullable

# Type alias for internal hint clarity
num = int | float
