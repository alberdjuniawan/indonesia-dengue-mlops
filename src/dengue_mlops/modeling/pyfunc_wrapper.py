from __future__ import annotations

from typing import Any

import mlflow.pyfunc

from dengue_mlops.modeling.models import BaseModel


class DengueMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc Wrapper ensuring universal inference compatibility."""

    def __init__(self, model: BaseModel, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names

    def predict(self, context: Any, model_input: Any) -> list[float]:
        # Handle Pandas DataFrame
        if hasattr(model_input, "values"):
            if hasattr(model_input, "columns"):
                available_cols = [c for c in self.feature_names if c in model_input.columns]
                if len(available_cols) == len(self.feature_names):
                    data = model_input[self.feature_names].values.tolist()
                else:
                    data = model_input.values.tolist()
            else:
                data = model_input.values.tolist()
        elif isinstance(model_input, list):
            data = model_input
        else:
            data = list(model_input)

        return self.model.predict(data)
