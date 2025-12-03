import requests
import os

def download_file_from_google_drive(file_id, destination):
    """
    Reliable Google Drive file downloader with confirm token handling.
    Works even for large files (100MB+).
    """

    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    # Initial request
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    # If there is a confirmation token, repeat request with token
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    """Extracts the confirmation token needed for large files."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def _save_response_content(response, destination):
    """Writes the binary response to file."""
    CHUNK_SIZE = 32768  # 32KB

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
