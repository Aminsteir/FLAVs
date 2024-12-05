import argparse
from torchviz import make_dot
import torch
from models.dual_stream import DualStreamModel
from models.spatio_temporal import SpatioTemporalModel

def make_diagram(model_type):
    if model_type == "dual_stream":
        # Dummy input matching your model's input shape
        frame_input = torch.randn(1, 9, 256, 455)  # Batch size of 1, 9 channels, HxW
        flow_input = torch.randn(1, 2, 256, 455)  # Batch size of 1, 2 channels, HxW
        model = DualStreamModel()
        output = model(frame_input, flow_input)
    elif model_type == "spatio_temporal":
        frame_input = torch.randn(1, 3, 3, 256, 455)  # Batch size of 1, 3 frames, 3 channels, HxW
        model = SpatioTemporalModel()
        output = model(frame_input)

    # Generate the computational graph
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = "png"  # Save as PNG for presentation
    dot.render(f"testing/outputs/{model_type}_architecture")  # Saves to '{model_type}_architecture.png'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model visualization")
    parser.add_argument("--model_type", type=str, default="dual_stream")

    args = parser.parse_args()

    make_diagram(args.model_type)
