# Configuration Management Protocol

This framework provides a centralized configuration management system for deep learning experiments. The system implements a modular configuration architecture that enables parametric control of model training, data processing, and optimization hyperparameters through structured configuration files.

### Configuration Factory Interface

```python
get_config(config_name: str) -> frozendict
```

**Parameters:**
- `config_name`: Registered configuration identifier

**Returns:** Immutable configuration dictionary

### Configuration Schema

Each configuration file must define a `CONFIG` dictionary with the following structure:

```python
CONFIG = {
    "model": (model_name: str, model_params: dict),
    "loss": (loss_name: str, loss_params: dict),
    "optimizer": (optimizer_name: str, optimizer_params: dict),
    "lr_scheduler": (scheduler_name: str, scheduler_params: dict),
    "train_dataloader": (loader_name: str, loader_params: dict),
    "val_dataloader": (loader_name: str, loader_params: dict),
    "train_transform": (transform_name: str, transform_params: dict),
    "test_transform": (transform_name: str, transform_params: dict),
    # Additional hyperparameters
    "max_epochs": int,
    "in_channels": int,
    "img_size": tuple,
    "seed": int,
    # ...
}
```

### Dynamic Parameter Support

The framework supports dynamic parameter computation through callable values:

```python
def max_iterations(trainer) -> int:
    return max_epochs * len(trainer.train_dataloader())

CONFIG = {
    "lr_scheduler": ("PolynomialLR", {
        "total_iters": max_iterations,  # Computed at runtime
    }),
}
```

## Implementation Requirements

1. **Module Structure**: Each configuration must be defined in a separate Python file
2. **Naming Convention**: Configuration files should follow the pattern `{model}_{dataset}.py`
3. **CONFIG Dictionary**: Must export a global `CONFIG` dictionary
4. **Immutable Returns**: All configuration access returns frozen dictionaries
5. **Auto-discovery**: Configurations are automatically discovered through module introspection

## Architecture Discovery

The framework automatically imports all Python modules in the configuration directory, enabling dynamic configuration registration without explicit imports.
