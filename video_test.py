import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as VF, InterpolationMode
from model import SegFormerHead, Dinov2ForSemanticSegmentation
import os
root = os.getcwd()

# =================================================================================================================================================

MODEL_NAME = "acpds_og"  # acpds_og | acpds | pklot
TEST_VIDEO = 2      # 1 | 2

MODEL = f"{root}/weights/model_{MODEL_NAME}.pt" 
video_path = f"{root}/test/test_{TEST_VIDEO}.mp4" 
save_path = f"{root}/test/output" 

# =================================================================================================================================================

ADE_MEAN = [0.485, 0.456, 0.406]  # Customize as needed
ADE_STD = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(MODEL, map_location=device, weights_only=False)
model.eval()
model.to(device)

id2color = {
    0: [0, 0, 0],          # background
    1: [144, 238, 144],    # light green
    2: [255, 160, 160],    # light red
    # Add more labels as needed
}

resize_size = (448, 448)
normalize = transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)

def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = VF.to_tensor(image)
    tensor = VF.resize(tensor, resize_size, interpolation=InterpolationMode.BILINEAR)
    normalized = normalize(tensor)
    return normalized.unsqueeze(0), image  # [1, C, H, W], original PIL image

def visualize_frame(original_image, segmentation_map):
    color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for label, color in id2color.items():
        color_seg[segmentation_map == label] = color
    overlay = np.array(original_image) * 0.5 + color_seg * 0.5
    return overlay.astype(np.uint8)

def predict_frame(frame_tensor, original_image):
    with torch.no_grad():
        outputs = model(frame_tensor.to(device))
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=original_image.size[::-1], mode="bilinear", align_corners=False
        )
        predicted_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        return visualize_frame(original_image, predicted_map)

def draw_polygons_on_frame(frame, predicted_map):
    color_map = {
        1: (0, 255, 0),     # green for unparked
        2: (0, 0, 255),     # red for parked
    }
    overlay = frame.copy()
    for cls in [1, 2]:
        mask = (predicted_map == cls).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_map[cls], thickness=2)
    return overlay

cap = cv2.VideoCapture(video_path)
target_fps = 2
original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(original_fps / target_fps)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

plt.ion()
fig, ax = plt.subplots()
img_display = None

frame_idx = 0
processed_count = 0
success = True
print("🔍 Processing video...")

out = cv2.VideoWriter(f"{save_path}/segmented_model_{MODEL_NAME}.mp4", fourcc, target_fps, (width, height))
polygon_out = cv2.VideoWriter(f"{save_path}/polygon_model_{MODEL_NAME}.mp4", fourcc, target_fps, (width, height))
combined_out = cv2.VideoWriter(f"{save_path}/combined_model_{MODEL_NAME}.mp4", fourcc, target_fps, (width*2, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if frame_idx % frame_interval == 0:
        frame_tensor, original_image = preprocess_frame(frame)
        with torch.no_grad():
            outputs = model(frame_tensor.to(device))
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=original_image.size[::-1], mode="bilinear", align_corners=False
            )
            predicted_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

        # Segmentation overlay
        seg_overlay = visualize_frame(original_image, predicted_map)
        seg_bgr = cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR)

        # Polygon overlay
        poly_overlay = draw_polygons_on_frame(frame, predicted_map)

        combined_frame = np.hstack((seg_bgr, poly_overlay))

        out.write(seg_bgr)
        polygon_out.write(poly_overlay)
        combined_out.write(combined_frame)

        if img_display is None:
            img_display = ax.imshow(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        else:
            img_display.set_data(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
        plt.pause(0.001)

        processed_count += 1

    frame_idx += 1

cap.release()
out.release()
polygon_out.release()
combined_out.release()
plt.ioff()
plt.close()
