import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import scipy.io


def download(url, dst):
    if not dst.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dst)
        print("Downloaded:", dst)
    else:
        print(f"File already exists: {dst}")


def extract(file, dst):
    if not dst.exists():
        print(f"Extracting {file} ...")
        with tarfile.open(file) as tar:
            tar.extractall(path=dst.parent)
        print("Extraction completed.")
    else:
        print(f"Folder already exists: {dst}")


def prepare_data():
    data_root = Path("data")
    data_raw = data_root / "data_raw"
    images_dir = data_raw / "jpg"

    data_root.mkdir(exist_ok=True)
    data_raw.mkdir(exist_ok=True)

    url_images = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    url_labels = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    url_setid = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

    tgz_path = data_raw / "102flowers.tgz"
    labels_path = data_raw / "imagelabels.mat"
    setid_path = data_raw / "setid.mat"

    # Download
    download(url_images, tgz_path)
    download(url_labels, labels_path)
    download(url_setid, setid_path)

    # Extract images
    extract(tgz_path, images_dir)

    # Load labels
    labels = scipy.io.loadmat(labels_path)["labels"][0]
    setid = scipy.io.loadmat(setid_path)

    train_ids = setid["trnid"][0] - 1
    val_ids = setid["valid"][0] - 1
    test_ids = setid["tstid"][0] - 1

    # Create folders
    for split in ["train", "val", "test"]:
        for cls in range(1, 103):
            (data_root / split / str(cls)).mkdir(parents=True, exist_ok=True)

    # All images
    all_images = sorted(images_dir.glob("*.jpg"))

    print(f"Total images detected: {len(all_images)}")

    print("Organizing images into train/val/test ...")
    for idx, img_path in enumerate(all_images):
        cls = labels[idx]

        if idx in train_ids:
            dst = data_root / "train" / str(cls) / img_path.name
        elif idx in val_ids:
            dst = data_root / "val" / str(cls) / img_path.name
        else:
            dst = data_root / "test" / str(cls) / img_path.name

        shutil.copy(img_path, dst)

    print("\n🎉 Dataset is ready! Located under /data/")
    print("train/, val/, test/ are now complete.")


if __name__ == "__main__":
    prepare_data()
