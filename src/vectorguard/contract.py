from typing import Any, Dict, get_type_hints
from vectorguard.fields import Field

class ContractMeta(type):
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        # Prevent processing the base class itself
        if name == "DataContract":
            return super().__new__(mcs, name, bases, namespace)

        fields: Dict[str, Field] = {}
        # Get annotations (type hints) like {'user_id': <class 'int'>}
        annotations = namespace.get("__annotations__", {})

        for field_name, expected_type in annotations.items():
            # Get the assigned Field instance or default to an empty one
            field_config = namespace.get(field_name)
            if not isinstance(field_config, Field):
                field_config = Field()
            
            # Store the expected datatype directly inside the configuration object
            field_config.data_type = expected_type
            fields[field_name] = field_config

        # Store processed contract fields in a protected attribute
        namespace["_contract_fields"] = fields
        return super().__new__(mcs, name, bases, namespace)


class DataContract(metaclass=ContractMeta):
    _contract_fields: Dict[str, Field] = {}

    @classmethod
    def get_fields(cls) -> Dict[str, Field]:
        """Returns the extracted dictionary of fields and constraints."""
        return cls._contract_fields
