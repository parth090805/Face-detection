from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import json
import os

SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

def authenticate_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

def is_image_file(mime_type):
    return mime_type.startswith("image/")

def get_flat_image_map(service, folder_id):
    file_map = {}

    def recurse(folder_id):
        page_token = None
        while True:
            response = service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()

            for file in response.get('files', []):
                file_id = file['id']
                name = file['name']
                mime_type = file['mimeType']

                if mime_type == 'application/vnd.google-apps.folder':
                    recurse(file_id)
                elif is_image_file(mime_type):
                    if name in file_map:
                        print(f"⚠️ Warning: Duplicate filename found: {name} (will overwrite previous entry)")
                    file_map[name] = file_id

            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

    recurse(folder_id)
    return file_map

if __name__ == '__main__':
    FOLDER_ID = "1H8G5bch-A5zZ3Bj8o7uQ41BZw_3wQhuO"  # Replace with your root folder ID

    service = authenticate_drive()
    image_map = get_flat_image_map(service, FOLDER_ID)

    with open("drive_file_map.json", "w") as f:
        json.dump(image_map, f, indent=4)

    print(f"✅ Generated drive_file_map.json with {len(image_map)} image entries.")
