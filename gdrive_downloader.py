import requests
import os

def extract_id(url: str):
    if "id=" in url:
        return url.split("id=")[1]
    if "/file/d/" in url:
        return url.split("/file/d/")[1].split("/")[0]
    raise ValueError("Invalid Google Drive URL")


def download_file_from_google_drive(url, destination):
    file_id = extract_id(url)
    download_url = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(download_url, params={"id": file_id}, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(
            download_url,
            params={"id": file_id, "confirm": token},
            stream=True,
        )

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"✔ Downloaded: {destination}")
