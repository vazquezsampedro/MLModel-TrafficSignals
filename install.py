pip install kaggle

import os

os.makedirs("/root/.kaggle", exist_ok=True)

dbutils.fs.cp("/FileStore/tables/kaggle.json", "file:/root/.kaggle/kaggle.json")

os.chmod("/root/.kaggle/kaggle.json", 600)

%pip install -q kaggle

!kaggle datasets download -d pkdarabi/cardetection -p /databricks/driver/cardetection --unzip

import os
data_dir = "/databricks/driver/cardetection"

print("Contenido del directorio")
for f in os.listdir(data_dir):
    print(f)

images_dir = os.path.join(data_dir, "car/test")
if os.path.exists(images_dir):
    print("\nClases encontradas:")
    for class_folder in os.listdir(images_dir):
        print(f"-{class_folder} ({len(os.listdir(os.path.join(images_dir, class_folder)))} imágenes)")

images_dir = os.path.join(data_dir, "car/train")
if os.path.exists(images_dir):
    print("\nClases encontradas:")
    for class_folder in os.listdir(images_dir):
        print(f"-{class_folder} ({len(os.listdir(os.path.join(images_dir, class_folder)))} imágenes)")

images_dir = os.path.join(data_dir, "car/valid")
if os.path.exists(images_dir):
    print("\nClases encontradas:")
    for class_folder in os.listdir(images_dir):
        print(f"-{class_folder} ({len(os.listdir(os.path.join(images_dir, class_folder)))} imágenes)")

