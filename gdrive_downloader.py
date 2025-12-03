import os
import re
import requests

def extract_file_id(url: str) -> str:
    """
    從 Google Drive URL 提取 file ID
    支援格式：
    - https://drive.google.com/file/d/1fe85t5UJhNYCCQBgpGSKCXnW4BcQvVkj/view?usp=sharing
    - https://drive.google.com/open?id=1fe85t5UJhNYCCQBgpGSKCXnW4BcQvVkj
    - https://drive.google.com/uc?id=1fe85t5UJhNYCCQBgpGSKCXnW4BcQvVkj
    """
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",   # /file/d/FILE_ID/
        r"id=([a-zA-Z0-9_-]+)"        # id=FILE_ID
    ]

    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)

    raise ValueError("無法解析 Google Drive file ID，請確認連結格式是否正確")


def download_file_from_google_drive(url: str, output_path: str):
    """
    從 Google Drive 下載檔案（自動處理確認頁面）
    """
    file_id = extract_file_id(url)
    download_url = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(download_url, params={'id': file_id}, stream=True)

    # 檢查是否需要 confirm token（Google 防毒警告頁面）
    token = get_confirm_token(response)
    if token:
        response = session.get(download_url, params={'id': file_id, 'confirm': token}, stream=True)

    # 建立目錄
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 寫出檔案
    save_response_content(response, output_path)

    return output_path


def get_confirm_token(response):
    """
    從 Google Drive 的 cookie 分析 confirm token
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination):
    """
    實際寫入檔案
    """
    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


# 測試用：放在底部避免 Streamlit import 時執行
if __name__ == "__main__":
    url = "https://drive.google.com/file/d/1fe85t5UJhNYCCQBgpGSKCXnW4BcQvVkj/view?usp=drive_link"
    output = "model/best_model.pt"
    print("Downloading...")
    download_file_from_google_drive(url, output)
    print("Download completed:", output)
