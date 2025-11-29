import os
import tarfile
import urllib.request
from sklearn.model_selection import train_test_split
import shutil

URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
LABEL_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

def download(url, path):
    if not os.path.exists(path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)

def extract(tgz, out_dir):
    print(f"Extracting to {out_dir}...")
    with tarfile.open(tgz) as tar:
        tar.extractall(out_dir)

def main():
    os.makedirs("data_raw", exist_ok=True)

    download(URL, "data_raw/102flowers.tgz")
    extract("data_raw/102flowers.tgz", "data_raw")

    imgs_dir = "data_raw/jpg"
    imgs = sorted(os.listdir(imgs_dir))

    # split
    train_files, test_files = train_test_split(imgs, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)

    def prepare_split(files, split):
        for fname in files:
            cls_id = int(fname.split("_")[0])  # Flower102 labels format
            out = f"data/{split}/{cls_id}"
            os.makedirs(out, exist_ok=True)
            shutil.copy(os.path.join(imgs_dir, fname), os.path.join(out, fname))

    prepare_split(train_files, "train")
    prepare_split(val_files, "val")
    prepare_split(test_files, "test")

    print("Data prepared under data/train, data/val, data/test")

if __name__ == "__main__":
    main()
