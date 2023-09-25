from enum import Enum, IntEnum
from typing import Any, ClassVar, Union, Optional, Iterator
from uuid import uuid4

import pandas as pd
import numpy as np

from pydantic import (
    ConfigDict,
    BaseModel,
    Field,
    model_validator,
    create_model,
    computed_field,
    UUID4,
    InstanceOf,
)
from pydantic.fields import FieldInfo

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
    space_id: ClassVar[UUID4] = uuid4()
    id: UUID4 = Field(..., default_factory=lambda: uuid4())

    @classmethod
    @property
    def features_list(cls) -> Iterator[tuple[str, FieldInfo]]:
        """
        Usage:

        ```
        for featurename, fieldinfo in self:
            yield feature, fieldinfo
        ```
        """
        return filter(lambda field: field[0] != "id", cls.model_fields.items())

    @classmethod
    @property
    def feature_names(cls) -> Iterator[str]:
        """
        Usage:

        ```
        for feature in self:
            featurename
        ```
        """
        return iter(name for name, _ in cls.features_list)

    @classmethod
    def to_featuremanager(cls, feature: str):
        feature_names = cls.feature_names
        if feature not in feature_names:
            raise ValueError(
                f"{feature} is not a valid feature; options: {list(feature_names)}"
            )

        if feature == "id":
            raise ValueError("ID is not a samplable feature.")

        feature_def = cls.model_json_schema()["properties"][feature]

        # TODO: Consider replacing with a protected accessor
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
        space = FeatureSpace(name=cls.__name__, id=cls.space_id)
        for param, field_info in cls.features_list:
            assert "id" != param
            space.features[param] = cls.to_featuremanager(param)

        return space

    @classmethod
    def sample_once(cls):
        data = {}
        for fieldname, _ in cls.features_list:
            feature = cls.to_featuremanager(fieldname)
            sample = feature.sample()
            data[feature.fieldname] = sample

        return cls(**data)

    def __iter__(self) -> Iterator[tuple[str, Union[float, IntEnum]]]:
        """
        Usage:

        ```
        for feature, value in self:
            yield feature, value
        ```
        """
        return iter(
            (
                (feature, getattr(self, feature))
                for (feature, feature_def) in self.features_list
            )
        )

    # should this be a series?
    def to_series(self):
        data = self.model_dump()
        series = pd.Series(data)
        return series

    @classmethod
    @property
    def index_cols(self, with_space_id=False):
        base = ["id"] if not with_space_id else ["space_id", "id"]
        base.extend(self.feature_names)
        return base


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
    id: UUID4 = Field(..., default_factory=lambda: uuid4())
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
            self.name,
            __base__=base,
            **field_definitions,
        )

    def __iter__(self):
        iter(self.features.items())

    def sample_df(self, n: int = 1000, with_ids=True):
        df = pd.DataFrame()
        for feature, feature_def in self.features.items():
            df[feature] = feature_def.sample(n)

        if with_ids:
            df["space_id"] = self.id.int
            df["id"] = [uuid4().int for _ in range(n)]

            metadata_cols = ["id", "space_id"]
            df.columns = pd.MultiIndex.from_tuples(
                [
                    ("metadata" if colname in metadata_cols else "features", colname)
                    for colname in df.columns
                ],
                names=["group", "field"],
            )
            df = df[["metadata", "features"]]

        return df

    def make_population(self, n: int = 1000, with_ids=True, index_by_metadata=True):
        return Population.from_feature_space(
            space=self, n=n, with_ids=with_ids, index_by_metadata=index_by_metadata
        )


class Population(BaseModel):
    model_config = ConfigDict(
        title="Design Space Population",
        arbitrary_types_allowed=True,
    )

    id: UUID4 = Field(..., default_factory=lambda: uuid4())
    space: FeatureSpace
    data: pd.DataFrame

    @computed_field
    @property
    def size(self) -> int:
        return len(self.data)

    @classmethod
    def from_feature_space(
        cls, space: FeatureSpace, n: int, with_ids=True, index_by_metadata=True
    ):
        if index_by_metadata and not with_ids:
            raise ValueError(
                "'with_ids' must be set to true to use the index_by_metadata option"
            )
        df = space.sample_df(n=n, with_ids=with_ids)
        pop = cls(space=space, data=df)
        if index_by_metadata:
            pop.index_by_metadata()
        return pop

    def index_by_metadata(self):
        # TODO: add check if it is already indexed this way
        self.data = self.data.set_index(
            keys=[("metadata", colname) for colname in self.data["metadata"].columns]
        )  # ["features"]
        self.data.index.names = (name for (group, name) in self.data.index.names)

    def to_index(self):
        pass
