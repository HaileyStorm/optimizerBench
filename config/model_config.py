from dataclasses import dataclass

@dataclass
class BaseModelConfig:
    input_dim: int
    output_dim: int
    hidden_dims: list
    activation: str
    dropout_rate: float

    def __post_init__(self):
        # Perform any necessary validation here
        assert self.input_dim > 0, "Input dimension must be positive"
        assert self.output_dim > 0, "Output dimension must be positive"
        assert all(dim > 0 for dim in self.hidden_dims), "Hidden dimensions must be positive"
        assert 0 <= self.dropout_rate <= 1, "Dropout rate must be between 0 and 1"
