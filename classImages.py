class CustomTrafficSignDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        image_files = sorted(os.listdir(images_dir))
        self.image_files = []

        for image_file in image_files:
            label_file = image_file.replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line and len(first_line.split()) > 0:
                        self.image_files.append(image_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        label_path = os.path.join(self.labels_dir, image_file.replace('.jpg', '.txt'))

        image = Image.open(image_path).convert("RGB")

        with open(label_path, 'r') as f:
            class_id = int(f.readline().strip().split()[0])

        if self.transform:
            image = self.transform(image)

        return image, class_id

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

base_path = "/databricks/driver/cardetection/car"

# Datasets
train_dataset = CustomTrafficSignDataset(
    images_dir=os.path.join(base_path, "train/images"),
    labels_dir=os.path.join(base_path, "train/labels"),
    transform=transform
)

valid_dataset = CustomTrafficSignDataset(
    images_dir=os.path.join(base_path, "valid/images"),
    labels_dir=os.path.join(base_path, "valid/labels"),
    transform=transform
)

test_dataset = CustomTrafficSignDataset(
    images_dir=os.path.join(base_path, "test/images"),
    labels_dir=os.path.join(base_path, "test/labels"),
    transform=transform
)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(set([label for _, label in train_dataset]))
class_names = [str(i) for i in range(num_classes)]
print(f"✔️ Dataset loaded: {num_classes} classes.")

