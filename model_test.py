from datasets import Dataset, DatasetDict, Image
import os
import random
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall, MulticlassPrecision
from torchmetrics import Accuracy, JaccardIndex, MeanMetric
from tqdm.auto import tqdm
from dataset import SegmentationDataset, collate_fn
from torch.utils.data import DataLoader
from model import SegFormerHead, Dinov2ForSemanticSegmentation
import torch
import matplotlib.pyplot as plt


# CONSTANTS ============================================================================================
MODEL_NAME = "acpds_og"  # acpds_og | acpds | pklot
dataset_name = "PKLOT" # ACPDS | PKLOT
train_ratio = 0.05
test_ratio = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3

root = os.getcwd()
MODEL = f"{root}/weights/model_{MODEL_NAME}.pt" 
dataset_root = f"{root}/{dataset_name}/{dataset_name}"
image_dir = os.path.join(dataset_root, "images")
mask_dir = os.path.join(dataset_root, "int_masks")

f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
# accuracy_metric = Accuracy(task='multiclass', num_classes=num_classes, top_k=2).to(device)
# mean_iou_metric = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)
loss_metric = MeanMetric().to(device)

id2label = {
    0: "background",
    1: "free",
    2: "occupied"
}

# ===================================================================================

def get_dataset():

    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    ])

    label_paths = sorted([
        os.path.join(mask_dir, f.replace(".jpg", ".png"))
        for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    ])

    # Shuffle together to preserve mapping
    combined = list(zip(image_paths, label_paths))
    random.seed(42)  # for reproducibility
    random.shuffle(combined)

    train_size = int(len(combined)*train_ratio)
    test_size = int(len(combined)*test_ratio)
    val_size = len(combined) - train_size - test_size

    train_split = combined[:train_size]
    test_split = combined[train_size:train_size + test_size]
    val_split = combined[train_size + test_size:]

    # Unzip back
    train_imgs, train_masks = zip(*train_split)
    test_imgs, test_masks = zip(*test_split)
    val_imgs, val_masks = zip(*val_split)

    def create_dataset(image_paths, label_paths):
        dataset = Dataset.from_dict({
            "image": image_paths,
            "label": label_paths
        })
        dataset = dataset.cast_column("image", Image())
        dataset = dataset.cast_column("label", Image())
        return dataset

    dataset = DatasetDict({
        "train": create_dataset(train_imgs, train_masks),
        "test": create_dataset(test_imgs, test_masks),
        "validation": create_dataset(val_imgs, val_masks)
    })

    # print(dataset)
    return dataset

model = torch.load(MODEL, map_location=device, weights_only=False)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = get_dataset()
test_dataset = SegmentationDataset(dataset["test"])
test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)

test_loss_list, test_iou_list, test_acc_list, test_f1_list = [], [], [], []
test_recall_list, test_precision_list = [], []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="🧪 Testing"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
    
        outputs = model(pixel_values, labels=labels)
        logits = outputs.logits
        loss = outputs.loss
        predicted = logits.argmax(dim=1)

        # print(f"{labels.shape=} {predicted.shape=}")

        f1_metric.update(predicted, labels)
        recall_metric.update(predicted, labels)
        precision_metric.update(predicted, labels)
        # accuracy_metric.update(predicted, labels)
        # mean_iou_metric.update(predicted, labels)
        loss_metric.update(loss)

f1 = f1_metric.compute()
recall = recall_metric.compute()
precision = precision_metric.compute()
# mean_iou = mean_iou_metric.compute()
# mean_accuracy = accuracy_metric.compute()
avg_loss = loss_metric.compute()

test_loss_list.append(avg_loss.item())
# test_iou_list.append(mean_iou.item())
# test_acc_list.append(mean_accuracy.item())
test_f1_list.append(f1.item())
test_recall_list.append(recall.item())
test_precision_list.append(precision.item())

print("\n🧪 Test Results:")
print(f"Loss: {avg_loss:.4f}")
# print(f"Mean IoU: {mean_iou:.4f}")
# print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
