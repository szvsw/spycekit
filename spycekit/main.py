from enum import Enum, IntEnum
from typing import Optional, Union, get_origin, get_args

import numpy as np

from pydantic import BaseModel, Field, model_validator, create_model, computed_field

# TODO: Pandera Integration
# TODO: saving and writing files
# TODO: versioning
# TODO: Example of merging/extending a design space while still benefitting from class methods
# potentially using passed in base model.
# TODO: mean/std and other sampling types
# TODO: normalization
# TODO: datatypes (tensor, tensorgpu, numpy dataframe, panderas, etc)
# TODO: disallow non int-enums
# TODO: allow string enums for types, as well as float enums
# TODO: add __iter__, __getitem__ methods
# TODO: add codegen utility


class DesignBase(BaseModel):
    @classmethod
    def to_featuremanager(cls, feature: str):
        if feature not in cls.model_fields:
            raise ValueError(
                f"{feature} is not a valid feature; options: {list(cls.model_fields.keys())}"
            )

        feature_def = cls.model_json_schema()["properties"][feature]
        feature_field = cls.model_fields[feature]
        if issubclass(feature_field.annotation, Enum):
            options = {
                enumval.name: enumval.value
                for enumval in feature_field.annotation._member_map_.values()
            }
            return FeatureManager(
                bounds=options,
                mode=FeatureType.Categorical,
                fieldname=feature,
            )
        else:
            return FeatureManager(
                bounds=(feature_def["minimum"], feature_def["maximum"]),
                mode=FeatureType.Continuous,
                fieldname=feature,
            )

    @classmethod
    def to_featurespace(cls):
        space = FeatureSpace(name=cls.__name__)
        for param in cls.model_fields:
            space.features[param] = cls.to_featuremanager(param)
        return space

    @classmethod
    def sample_once(cls):
        data = {}
        for fieldname in cls.model_fields:
            feature = cls.to_featuremanager(fieldname)
            sample = feature.sample()
            data[feature.fieldname] = sample

        return cls(**data)


class FeatureType(str, Enum):
    Categorical = "Categorical"
    Continuous = "Continuous"


class FeatureManager(BaseModel):
    fieldname: str
    bounds: Union[
        dict[str, int], tuple[float, float]
    ]  # tuple when enum, float when continuous
    mode: FeatureType

    @computed_field
    @property
    def min(self) -> float:
        if self.mode != FeatureType.Continuous:
            return None
        elif self.mode == FeatureType.Continuous:
            return self.bounds[0]

    @computed_field
    @property
    def max(self) -> float:
        if self.mode == FeatureType.Categorical:
            return None
        elif self.mode == FeatureType.Continuous:
            return self.bounds[1]

    def sample(self, n: Optional[int] = None):
        if n is None:
            if self.mode is FeatureType.Continuous:
                return (
                    np.random.rand() * (self.bounds[1] - self.bounds[0])
                    + self.bounds[0]
                )
            else:
                return np.random.randint(len(self.bounds))
        else:
            if self.mode is FeatureType.Continuous:
                return (
                    np.random.rand(n) * (self.bounds[1] - self.bounds[0])
                    + self.bounds[0]
                )
            else:
                return np.random.randint(len(self.bounds), size=n)

    @model_validator(mode="after")
    def check(self):
        if self.mode == FeatureType.Continuous:
            if not isinstance(self.bounds, tuple):
                raise ValueError(
                    f"In Continuous mode, feature bounds must be a tuple of floats representing min and max, not '{type(self.bounds), self.bounds}'"
                )
            if (
                not all((type(bound) == float for bound in self.bounds))
                or len(self.bounds) != 2
            ):
                raise ValueError(
                    f"In Continuous mode, feature bounds must be a tuple of floats reprsenting min and max, not '{type(self.bounds), self.bounds}'"
                )
            self.bounds = (min(self.bounds), max(self.bounds))
        elif self.mode == FeatureType.Categorical:
            if not isinstance(self.bounds, dict):
                raise ValueError(
                    f"In Categorical mode, feature bounds must be a dict[str,int] to make an enum, not '{type(self.bounds), self.bounds}'"
                )
            if not all((isinstance(key, str) for key in self.bounds.keys())) or not all(
                isinstance(val, int) for val in self.bounds.values()
            ):
                raise ValueError(
                    f"In Categorical mode, feature bounds must be a dict[str,int] to make an enum, not '{type(self.bounds), self.bounds}'"
                )


class FeatureSpace(BaseModel):
    name: str
    features: dict[str, FeatureManager] = {}

    def to_designmodel(self, base=DesignBase):
        field_definitions = {}
        for fieldname, fielddef in self.features.items():
            if fielddef.mode == FeatureType.Categorical:
                fieldclass = IntEnum(
                    fieldname, {key: val for key, val in fielddef.bounds.items()}
                )
                field_definitions[fieldname] = (fieldclass, Field(...))
            elif fielddef.mode == FeatureType.Continuous:
                field_definitions[fieldname] = (
                    float,
                    Field(..., ge=fielddef.min, le=fielddef.max),
                )
            else:
                raise ValueError(
                    f"{fieldname} is set to an unknown FeatureType {fielddef.mode}"
                )

        # TODO: feature spaces should have names, and that name should be used for design model
        return create_model(
            "Design",
            __base__=base,
            **field_definitions,
        )


if __name__ == "__main__":
    import json

    class Style(IntEnum):
        Cool = 0
        Fast = 1

    class MyDesign(DesignBase):
        length: float = Field(..., ge=0.5, le=2.5)
        width: float = Field(..., ge=3.5, le=4.5)
        style: Style

    sim = MyDesign.sample_once()

    # Similar, but the latter can be used with datamodel-codegen
    # Option 1: Convert the Design to a FeatureSpace, then dump the feature space
    featurespace = MyDesign.to_featurespace()
    serialized_featurespace = featurespace.model_dump_json(indent=4)
    print(serialized_featurespace)

    # Option 2:
    # Dump the JSON Schema of the design
    serialized_model_schema = json.dumps(MyDesign.model_json_schema(), indent=4)

    # Deserialization
    # Option 1
    # Reconstruct a class which is identical to the design (up to validation) from a feature space representation!
    # Assume you have already loaded the featurespace using standard pydantic deserializer
    # NB: Option 2 would be using json schema and data model codegen
    NewModel = featurespace.to_designmodel()
    d = NewModel(length=1, width=4, style=Style.Cool)
    d.style = Style.Fast
    print(NewModel.sample_once())
