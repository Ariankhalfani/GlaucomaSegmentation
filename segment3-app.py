import gradio as gr
from PIL import Image
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Assuming the model is loaded elsewhere
model = YOLO('best.pt')

def calculate_area(mask):
    return np.sum(mask > 0.5)  # Count true pixels in binary mask

# Your existing predict_image function refactored for Gradio
def predict_and_visualize(image):
    pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
    orig_size = pil_image.size
    results = model(pil_image)

    cup_area, disk_area, rim_area, rim_to_disk_ratio = 0, 0, 0, 0

    if len(results) > 0:
        result = results[0]

        if result.masks is not None and len(result.masks) > 0:
            # Convert the mask to a numpy array and apply the threshold
            disk_mask = np.array(result.masks[0].data.cpu().squeeze().numpy())
            disk_area = calculate_area(disk_mask > 0.5)

        if len(result.masks) > 1:
            # Convert the mask to a numpy array and apply the threshold
            cup_mask = np.array(result.masks[1].data.cpu().squeeze().numpy())
            cup_area = calculate_area(cup_mask > 0.5)
            rim_area = disk_area - cup_area

        if disk_area > 0:
            rim_to_disk_ratio = rim_area / disk_area

        # Overlay the masks onto the original image
        combined_image = np.array(pil_image)
        if disk_area > 0:
            disk_mask_resized = np.array(Image.fromarray(disk_mask).resize(orig_size, Image.NEAREST))
            combined_image[disk_mask_resized > 0.5] = [255, 0, 0]  # Red for disk
        if cup_area > 0:
            cup_mask_resized = np.array(Image.fromarray(cup_mask).resize(orig_size, Image.NEAREST))
            combined_image[cup_mask_resized > 0.5] = [0, 0, 255]  # Blue for cup

        # Return the visualized image and the text results as a dictionary
        return combined_image, f"Cup area: {cup_area} pixels", f"Disk area: {disk_area} pixels", f"Rim area: {rim_area} pixels", f"Rim/Disk ratio: {rim_to_disk_ratio:.2f}"

    return None, "No results", "No results", "No results", "No results"


# Setup Gradio interface
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.inputs.Image(),
    outputs=[
        gr.outputs.Image(type="numpy", label="Segmented Image"),
        gr.outputs.Textbox(label="Cup Area"),
        gr.outputs.Textbox(label="Disk Area"),
        gr.outputs.Textbox(label="Rim Area"),
        gr.outputs.Textbox(label="Rim/Disk Ratio")
    ],
    title="Glaucoma Detection with YOLO",
    description="This app uses a YOLO model to detect and segment the optical cup and disc from retinal images."
)

if __name__ == "__main__":
    iface.launch()
