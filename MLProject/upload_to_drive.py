from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def upload_file_to_drive(local_path, drive_folder_id, drive_filename):
    """
    Upload file ke Google Drive folder tertentu.
    """
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile("MLProject/client_secrets.json")
    gauth.LocalWebserverAuth()  # login pertama kali via browser
    drive = GoogleDrive(gauth)

    file = drive.CreateFile({
        'title': drive_filename,
        'parents': [{'id': drive_folder_id}]
    })
    file.SetContentFile(local_path)
    file.Upload()
    print(f"✅ Uploaded {drive_filename} to Google Drive")

if __name__ == "__main__":
    # Ganti dengan ID folder Google Drive tujuan
    DRIVE_FOLDER_ID = "YOUR_FOLDER_ID"

    # Artefak yang mau diupload
    artifacts = [
        ("MLProject/confusion_matrix.png", "confusion_matrix.png"),
        ("MLProject/test_sample.csv", "test_sample.csv")
    ]

    for local_path, filename in artifacts:
        if os.path.exists(local_path):
            upload_file_to_drive(local_path, DRIVE_FOLDER_ID, filename)
        else:
            print(f"⚠️ File {local_path} tidak ditemukan")
