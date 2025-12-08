# Model Registry Protocol

This protocol provides a unified interface for instantiating neural network architectures through a registry-based model factory pattern. The system enables dynamic model creation and supports extensible architecture definitions.

### Model Factory Interface

```python
build_model(name: str, in_channels: int, num_classes: int, **kwargs) -> nn.Module
```

**Parameters:**
- `name`: Registered model identifier
- `in_channels`: Input tensor channel dimension
- `num_classes`: Output classification dimension
- `**kwargs`: Model-specific hyperparameters

**Returns:** Instantiated PyTorch model

### Registration Protocol

Models are registered using the `@register_model` decorator:

```python
@register_model("model_name")
class ModelClass(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        # Implementation
```

**Registration Variants:**
- `@register_model("name")`: Direct name assignment
- `@register_model(alias="name")`: Named parameter assignment

### Implementation Requirements

The framework automatically discovers model definitions through AST parsing of `__init__.py` files in model subdirectories, enabling dynamic registration without explicit imports.

1. **Module Structure**: Each model must reside in a dedicated package under the `model/` directory
2. **Class Interface**: All models must inherit from `torch.nn.Module` 
3. **Constructor Signature**: Must accept `in_channels`, `num_classes`, and optional `**kwargs`
4. **Registration**: Must be decorated with `@register_model`
