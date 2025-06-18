# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# import os
# import re
# import sys

# # Setup Google Drive authentication
# gauth = GoogleAuth()
# gauth.LocalWebserverAuth()
# drive = GoogleDrive(gauth)

# # === UTILITY FUNCTIONS ===

# def extract_folder_id(link):
#     match = re.search(r'/folders/([a-zA-Z0-9_-]+)', link)
#     if not match:
#         raise ValueError("Invalid Google Drive folder link.")
#     return match.group(1)

# def download_images_recursively(folder_id, save_path):
#     os.makedirs(save_path, exist_ok=True)

#     # Get all files and folders in this folder
#     file_list = drive.ListFile({
#         'q': f"'{folder_id}' in parents and trashed=false"
#     }).GetList()

#     for file in file_list:
#         title = file['title']
#         mime_type = file['mimeType']
#         file_id = file['id']

#         if mime_type == 'application/vnd.google-apps.folder':
#             # Recursively process subfolders
#             print(f"📁 Entering folder: {title}")
#             subfolder_path = os.path.join(save_path, title)
#             download_images_recursively(file_id, subfolder_path)
#         elif title.lower().endswith(('.jpg', '.jpeg', '.png')):
#             print(f"⬇️ Downloading: {title}")
#             file.GetContentFile(os.path.join(save_path, title))


# # === MAIN ===

# if __name__ == "__main__":
#     # Accept folder link from user
#     if len(sys.argv) < 2:
#         print("Usage: python download.py <Google Drive Folder Link>")
#         sys.exit(1)

#     shared_folder_link = sys.argv[1]
#     try:
#         root_folder_id = extract_folder_id(shared_folder_link)
#     except ValueError as e:
#         print(f"❌ {e}")
#         sys.exit(1)

#     # Start recursive download
#     download_dir = "downloaded_images"
#     print("🚀 Starting download...")
#     download_images_recursively(root_folder_id, download_dir)
#     print("✅ All images downloaded.")

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import re
import sys
import time
import socket
from httplib2 import HttpLib2Error

# Setup Google Drive authentication
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# === UTILITY FUNCTIONS ===

def extract_folder_id(link):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', link)
    if not match:
        raise ValueError("Invalid Google Drive folder link.")
    return match.group(1)

def safe_download(file, path, retries=3, delay=5):
    for attempt in range(1, retries + 1):
        try:
            file.GetContentFile(path)
            return True
        except (HttpLib2Error, socket.timeout, TimeoutError) as e:
            print(f"⚠️  Download failed ({attempt}/{retries}) for {file['title']}: {e}")
            time.sleep(delay)
    print(f"❌ Skipped: {file['title']} after {retries} retries.")
    return False

# def download_images_recursively(folder_id, save_path):
#     os.makedirs(save_path, exist_ok=True)

#     # Get all files and folders in this folder
#     file_list = drive.ListFile({
#         'q': f"'{folder_id}' in parents and trashed=false"
#     }).GetList()

#     for file in file_list:
#         title = file['title']
#         mime_type = file['mimeType']
#         file_id = file['id']

#         if mime_type == 'application/vnd.google-apps.folder':
#             print(f"📁 Entering folder: {title}")
#             subfolder_path = os.path.join(save_path, title)
#             download_images_recursively(file_id, subfolder_path)

#         elif title.lower().endswith(('.jpg', '.jpeg', '.png')):
#             print(f"⬇️ Downloading: {title}")
#             target_path = os.path.join(save_path, title)
#             success = safe_download(file, target_path)
#             if not success:
#                 with open("failed_downloads.log", "a") as log:
#                     log.write(f"{file_id}\t{title}\n")

def download_images_recursively(folder_id, save_path):
    os.makedirs(save_path, exist_ok=True)

    # Get all files and folders in this folder
    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()

    for file in file_list:
        title = file['title']
        mime_type = file['mimeType']
        file_id = file['id']

        if mime_type == 'application/vnd.google-apps.folder':
            print(f"📁 Entering folder: {title}")
            subfolder_path = os.path.join(save_path, title)
            download_images_recursively(file_id, subfolder_path)

        elif title.lower().endswith(('.jpg', '.jpeg', '.png')):
            target_path = os.path.join(save_path, title)
            if os.path.exists(target_path):
                print(f"⏭️ Skipping already downloaded: {title}")
                continue
            print(f"⬇️ Downloading: {title}")
            success = safe_download(file, target_path)
            if not success:
                with open("failed_downloads.log", "a") as log:
                    log.write(f"{file_id}\t{title}\n")


# === MAIN ===

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download.py <Google Drive Folder Link>")
        sys.exit(1)

    shared_folder_link = sys.argv[1]
    try:
        root_folder_id = extract_folder_id(shared_folder_link)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    download_dir = "downloaded_images"
    print("🚀 Starting download...")
    download_images_recursively(root_folder_id, download_dir)
    print("✅ All reachable images downloaded.")
