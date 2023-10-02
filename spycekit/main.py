from enum import Enum, IntEnum
from typing import Any, ClassVar, Union, Optional, Iterator, Literal
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


PopIndexType = Literal["metadata", "all", "features", None]


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
        return iter(self.features.items())

    def transform_features(self, features: pd.DataFrame):
        # create a new df to hold the transformed features
        features_transformed = pd.DataFrame(index=features.index)

        # iterate over the features in the feature space
        for feature, feature_def in self:
            # if the feature is categorical, use pd.get_dummies to one-hot encode
            if feature_def.mode == FeatureType.Categorical:
                feature_vals = features[feature]
                onehots = pd.get_dummies(
                    feature_vals, prefix=feature, prefix_sep="_", dtype=bool
                )
                features_transformed = features_transformed.join(onehots)
            # if the feature is continuous, use the bounds to normalize to [0,1]
            elif feature_def.mode == FeatureType.Continuous:
                fmin, fmax = feature_def.bounds
                features_transformed[feature] = (features[feature] - fmin) / (
                    fmax - fmin
                )
        return features_transformed

    def inverse_transform_features(self, features: pd.DataFrame):
        features_untransformed = pd.DataFrame(index=features.index)

        # iterate over the features in the feature space
        for feature, feature_def in self:
            # if the feature is categorical, we get the onehot columns and then use the column which is true
            if feature_def.mode == FeatureType.Categorical:
                feat_names = features.columns[features.columns.str.startswith(feature)]
                catcodes = pd.Series(
                    [int(feat_name.split("_")[-1]) for feat_name in feat_names]
                )
                # get the feature columns
                feat_vals = features[feat_names]
                # get the column which is true for each row
                idx = feat_vals.idxmax(axis=1)
                # split the index string and get the last element, which is the category
                idx = (idx.str.split("_").str[-1]).astype(int)
                features_untransformed[feature] = idx

            # if the feature is continuous, use the bounds to normalize to [0,1]
            elif feature_def.mode == FeatureType.Continuous:
                fmin, fmax = feature_def.bounds
                features_untransformed[feature] = (features[feature]) * (
                    fmax - fmin
                ) + fmin
        return features_untransformed

    def sample_df(self, n: int = 1000):
        df = pd.DataFrame()
        for feature, feature_def in self.features.items():
            df[feature] = feature_def.sample(n)

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

    def make_population(
        self,
        n: int = 1000,
        index_by: PopIndexType = None,
        flatten=False,
    ):
        return Population.from_feature_space(
            space=self,
            n=n,
            index_by=index_by,
            flatten=flatten,
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
        cls, space: FeatureSpace, n: int, index_by: PopIndexType = None, flatten=False
    ):
        df = space.sample_df(n=n)
        pop = cls(space=space, data=df)
        pop.index_by(index_by, flatten=flatten)
        return pop

    def index_by(
        self,
        index_by: PopIndexType = None,
        flatten=False,
    ):
        # TODO: add check if it is already indexed this way / unset index
        if index_by in ["metadata", "features"]:
            group_to_take = index_by
            self.data = self.data.set_index(
                keys=[
                    (group_to_take, colname)
                    for colname in self.data[group_to_take].columns
                ]
            )
            if flatten:
                group_to_leave = "metadata" if index_by == "features" else "features"
                self.data = self.data[group_to_leave]
                self.data.index.names = (
                    name for (group, name) in self.data.index.names
                )
        elif index_by == "all":
            # From  hierarchical to flat columns
            self.data = self.data.set_index(
                keys=list(self.data.columns.to_flat_index().values)
            )
            self.data = pd.DataFrame(index=self.data.index)
            if flatten:
                self.data.index.names = (
                    name for (group, name) in self.data.index.names
                )
        elif index_by == None:
            if flatten:
                self.data.columns = self.data.columns.to_flat_index()

                self.data = self.data.rename(
                    mapper={(group, name): name for group, name in self.data.columns},
                    axis=1,
                )

    def to_index(self):
        pass
