from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as VF
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

class SegmentationDataset(Dataset):
    def __init__(self, dataset, resize_size=(448, 448)):
        self.dataset = dataset
        self.resize_size = resize_size
        self.normalize = transforms.Normalize(mean=ADE_MEAN, std=ADE_STD)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        og_image = np.array(item["image"])
        og_mask = np.array(item["label"])

        # Convert image to tensor
        image = VF.to_tensor(og_image)  # [C, H, W], scaled to [0, 1]
        mask = torch.from_numpy(og_mask).long()  # [H, W] with class IDs

        # Resize
        image = VF.resize(image, self.resize_size, interpolation=InterpolationMode.BILINEAR)
        mask = VF.resize(mask.unsqueeze(0), self.resize_size, interpolation=InterpolationMode.NEAREST).squeeze(0)

        # Normalize
        image = self.normalize(image)

        return image, mask, og_image, og_mask
    
def collate_fn(inputs):
    batch = dict()
    batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
    batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
    batch["original_images"] = [i[2] for i in inputs]
    batch["original_segmentation_maps"] = [i[3] for i in inputs]

    return batch