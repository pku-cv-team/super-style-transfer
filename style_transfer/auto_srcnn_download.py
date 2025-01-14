import os
import requests

file_urls = [
    "https://www.dropbox.com/scl/fi/9608kumo7yyg6mzqzvns2/srcnn_x2.pth?rlkey=ff6njqtgpanoywi2rngmz5ilh&e=1&dl=1",
    "https://www.dropbox.com/scl/fi/sx03zansjfjdxuftkxrwf/srcnn_x3.pth?rlkey=0tr8h4r4344xgdosy4zo7ce6l&e=1&dl=1",
    "https://www.dropbox.com/scl/fi/jv942327mse39x3s2wwe3/srcnn_x4.pth?rlkey=pbmizpptooiz0jgm2jtr92cri&e=1&dl=1"
]

download_dir = os.path.dirname(__file__)
os.makedirs(download_dir, exist_ok=True)

for url in file_urls:
    file_name = url.split("/")[-1].split("?")[0]
    file_path = os.path.join(download_dir, file_name)
    print(f"Downloading {file_name}...")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"{file_name} done")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_name}: {e}")