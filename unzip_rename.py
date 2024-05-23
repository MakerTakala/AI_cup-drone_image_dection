import zipfile
import os

zip_dir = "./Dataset/zipdata/"
train_data_zip_path = f"{zip_dir}35_Competition 2_Training dataset_V3.zip"
public_test_data_zip_path = f"{zip_dir}35_Competition 2_public testing dataset.zip"
private_test_data_zip_path = f"{zip_dir}35_Competition 2_Private Test Dataset.zip"
target_dir = "./Dataset/"

with zipfile.ZipFile(train_data_zip_path, "r") as zip_ref:
    zip_ref.extractall(target_dir)
with zipfile.ZipFile(public_test_data_zip_path, "r") as zip_ref:
    zip_ref.extractall(target_dir)
with zipfile.ZipFile(private_test_data_zip_path, "r") as zip_ref:
    zip_ref.extractall(target_dir)

os.rename(f"{target_dir}Training_dataset", f"{target_dir}Train_data_label")
os.rename(f"{target_dir}img", f"{target_dir}Public_test_data")
os.rename(
    f"{target_dir}35_Competition 2_Private Test Dataset/img/",
    f"{target_dir}Private_test_data",
)
os.removedirs(f"{target_dir}35_Competition 2_Private Test Dataset")
