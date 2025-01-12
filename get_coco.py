"""下载COCO数据集并解压缩，由Kimi生成"""

# pylint: disable=all

import os
import requests
from zipfile import ZipFile

# 创建 data/coco 目录
os.makedirs("data/coco", exist_ok=True)

# 下载文件的 URL 列表
urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
]

# 下载文件
for url in urls:
    filename = os.path.basename(url)
    filepath = os.path.join("data/coco", filename)
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

# 解压 zip 文件
zip_files = [f for f in os.listdir("data/coco") if f.endswith(".zip")]
for zip_file in zip_files:
    filepath = os.path.join("data/coco", zip_file)
    print(f"Unzipping {zip_file}...")
    with ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall("data/coco")
    print(f"Unzipped {zip_file}")

# 删除 zip 文件
for zip_file in zip_files:
    filepath = os.path.join("data/coco", zip_file)
    os.remove(filepath)
    print(f"Deleted {zip_file}")
