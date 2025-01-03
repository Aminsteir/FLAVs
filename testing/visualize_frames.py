import os
import numpy as np
from PIL import Image

from utils.optical_flow import compute_optical_flow

def vis_optical_flow():
    base_image_folder = 'data/base_model_training/data/'
    images = ['18121.jpg', '18122.jpg', '18123.jpg']
    paths = [os.path.join(base_image_folder, img) for img in images]
    frames = [Image.open(path).convert("RGB") for path in paths]
    flows = [compute_optical_flow(frames[i], frames[i+1]) for i in range(2)]
    save_dir = 'testing/outputs/optical_flows/'
    os.makedirs(save_dir, exist_ok=True)

    # Save each flow as a grayscale image
    for i, flow in enumerate(flows):
        # Convert tensor to numpy array
        flow_np = flow.cpu().numpy()

        # Normalize flow values to [0, 255] for visualization
        flow_normalized = (flow_np * 255).astype(np.uint8)

        # Save the normalized flow as an image
        flow_image = Image.fromarray(flow_normalized)
        flow_image.save(os.path.join(save_dir, f'flow_{images[i].split(".")[0]}.png'))

    print(f"Optical flow images saved to {save_dir}")

if __name__ == '__main__':
    vis_optical_flow()
