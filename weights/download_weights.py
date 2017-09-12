"""
This scripts will download all necessary weight files for this project

Reference:
https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
"""
import requests


def _main_():
  # Download DarkNet
  download_file_from_google_drive('https://drive.google.com/open?id=0ByoFGh573uhzMWZMb3J6OXdOemc','./darknet19.h5')


def download_file_from_google_drive(id, destination):
  def get_confirm_token(response):
    for key, value in response.cookies.items():
      if key.startswith('download_warning'):
        return value
    return None

  def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, 'wb') as f:
      for chunk in response.iter_content(CHUNK_SIZE):
        if chunk:  # filter out keep-alive new chunks
          f.write(chunk)

  URL = "https://docs.google.com/uc?export=download"
  session = requests.Session()

  response = session.get(URL, params={'id': id}, stream=True)
  token = get_confirm_token(response)

  if token:
    params = {'id': id, 'confirm': token}
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, destination)

if __name__ =='__main__':
  _main_()