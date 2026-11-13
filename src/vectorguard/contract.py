import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from vectorguard.fields import Field

SUPPORTED_TYPES = (int, float, str)


def _resolve_field_type(annotation: Any, field_name: str) -> tuple[type, bool]:
    """
    Resolves a Python type annotation into a supported primitive type and a
    boolean indicating whether None/null is permitted by the annotation.
    """
    origin = get_origin(annotation)

    # Handle Optional[T] or Union[T, None] or T | None
    if origin is Union or origin is types.UnionType:
        args = get_args(annotation)
        non_none_args = [arg for arg in args if arg is not type(None)]
        has_none = len(non_none_args) != len(args)

        if len(non_none_args) != 1:
            raise TypeError(
                f"Unsupported Union annotation for field '{field_name}': {annotation}. "
                f"Only Optional[T] / T | None with a single supported type is allowed."
            )

        base_type = non_none_args[0]
        if base_type not in SUPPORTED_TYPES:
            raise TypeError(
                f"Unsupported type '{base_type}' for field '{field_name}'. "
                f"Supported types are: {[t.__name__ for t in SUPPORTED_TYPES]}."
            )
        return base_type, has_none

    if annotation not in SUPPORTED_TYPES:
        raise TypeError(
            f"Unsupported type annotation '{annotation}' for field '{field_name}'. "
            f"Supported types are: {[t.__name__ for t in SUPPORTED_TYPES]}."
        )

    return annotation, False


class ContractMeta(type):
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        cls = super().__new__(mcs, name, bases, namespace)

        # Prevent processing the base class itself
        if name == "DataContract" and not bases:
            return cls

        fields: dict[str, Field] = {}

        # 1. Collect inherited fields from base contracts in MRO order (excluding object/DataContract)
        for base in reversed(cls.__mro__[1:]):
            base_fields = getattr(base, "_contract_fields", None)
            if isinstance(base_fields, dict):
                for field_name, base_field in base_fields.items():
                    # Clone inherited fields so subclass mutations never affect parent contracts
                    fields[field_name] = base_field.clone_with_type(
                        data_type=base_field.data_type,
                        force_nullable=base_field.nullable,
                    )

        # 2. Resolve type annotations declared directly on this class
        # Using get_type_hints resolves stringified annotations (from __future__ import annotations)
        try:
            resolved_hints = get_type_hints(cls)
        except NameError:
            # Fallback for locally-scoped test classes referencing local variables
            resolved_hints = cls.__dict__.get("__annotations__", {})

        raw_annotations = cls.__dict__.get("__annotations__", {})

        for field_name in raw_annotations:
            expected_annotation = resolved_hints.get(field_name, raw_annotations[field_name])
            resolved_type, is_optional = _resolve_field_type(expected_annotation, field_name)

            # Get assigned Field instance from class namespace or inherit rules if overriding type only
            field_config = namespace.get(field_name)
            if not isinstance(field_config, Field):
                if field_name in fields:
                    field_config = fields[field_name]
                else:
                    field_config = Field()

            # Clone the field to guarantee immutability across shared Field instances
            fields[field_name] = field_config.clone_with_type(
                data_type=resolved_type,
                force_nullable=is_optional,
            )

        cls._contract_fields = fields
        return cls


class DataContract(metaclass=ContractMeta):
    _contract_fields: dict[str, Field] = {}

    @classmethod
    def get_fields(cls) -> dict[str, Field]:
        """Returns a shallow copy of the extracted dictionary of fields and constraints."""
        return dict(cls._contract_fields)
