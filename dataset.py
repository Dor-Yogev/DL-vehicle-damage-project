import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import transforms


label_to_cls={0:"crack",
              1:"scratch",
              2:"tire flat",
              3:"dent",
              4:"glass shatter",
              5:"lamp broken"}

cls_to_label={"crack":0,
              "scratch":1,
              "tire flat":2,
              "dent":3,
              "glass shatter":4,
              "lamp broken":5}


class VehicleDamageDataset(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = f"./dataset/train/images/{self.filenames[idx]}"
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

def get_datasets(transform=None, image_dims=(224, 224)):
    dataset_df = pd.read_csv("./dataset/train/train.csv")
    dataset_df["label"] = dataset_df["label"] - 1
    train_val_df, test_df = train_test_split(dataset_df, test_size=0.135, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.16, random_state=42)

    X_train = train_df["filename"].tolist()
    Y_train = train_df["label"].tolist()
    X_val = val_df["filename"].tolist()
    Y_val = val_df["label"].tolist()
    X_test = test_df["filename"].tolist()
    Y_test = test_df["label"].tolist()

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_dims),
            transforms.ToTensor(),
        ])


    train_dataset = VehicleDamageDataset(X_train, Y_train, transform=transform)
    val_dataset = VehicleDamageDataset(X_val, Y_val, transform=transform)
    test_dataset = VehicleDamageDataset(X_test, Y_test, transform=transform)

    return train_dataset, val_dataset, test_dataset


def print_dataset_distribution(dataset, dataset_name):
    # Step 1: Extract labels
    labels = dataset.labels
    # Step 2: Count labels
    label_counts = Counter(labels)
    # Step 3: Plot the distribution
    labels, counts = zip(*label_counts.items())
    labels_names = [label_to_cls[i] for i in labels]
    plt.bar(labels_names, counts)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title(dataset_name)
    plt.show()


import matplotlib.pyplot as plt


def display_sample_images(dataset):
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    class_samples = {}

    for image, label in dataset:
        if label not in class_samples:
            class_samples[label] = image
        if len(class_samples) == 6:
            break

    for i in range(6):
        ax = axes[i]
        image = class_samples[i].numpy().transpose((1, 2, 0))
        ax.imshow(image)
        ax.set_title(f"{label_to_cls[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
