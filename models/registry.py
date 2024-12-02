from models.dual_stream import DualStreamModel
from models.spatio_temporal import SpatioTemporalModel
from models.temporal_transformer import TemporalTransformer

# Registry of available models
MODEL_REGISTRY = {
    "dual_stream": DualStreamModel,
    "spatio_temporal": SpatioTemporalModel,
    "temporal_transformer": TemporalTransformer,
}

def get_model(model_type, output_type="angle", **kwargs):
    """
    Retrieve the model class from the registry.

    Args:
        model_type (str): Name of the model type.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        nn.Module: The initialized model.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type](output_type=output_type, **kwargs)
