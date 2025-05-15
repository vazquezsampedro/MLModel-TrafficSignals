# 📦 Data Branch - Traffic Sign Classification

https://www.kaggle.com/datasets/pkdarabi/cardetection

This branch contains the **data folder structure** used for training and validating the traffic sign classification model.

## 📁 Folder Structure

- `train/images/` — Training images  
- `train/labels/` — Corresponding YOLO-format label files for training  
- `valid/images/` — Validation images  
- `valid/labels/` — Label files for validation  
- `test/images/` — Test images  
- `test/labels/` — Label files for testing  
- `sample_labels/` — Example `.txt` annotation files for reference  
- `sample_images/` — Example `.png` or `.jpg` images for preview/testing  

## ⚠️ Note

- Full image datasets are not included due to GitHub file size limits.
- You can:
  - Use [Git LFS](https://git-lfs.github.com/) for large files
  - Store datasets in cloud storage (e.g., Google Drive, AWS S3, Azure Blob) and load them dynamically.

## 🧾 Label Format (YOLO)

Each `.txt` label file contains one or more lines like:

