import argparse
import glob
import os
import requests
import sys

from requests import Response

COLLECTION_ID = '3917698'
FIGSHARE_BASE_URL = 'https://api.figshare.com/v2/'
DESTINATION_FOLDER = './datasets'

def get_collection_articles(collection_id: str = COLLECTION_ID, filter: str | None = None, page_size: int = 80) -> dict:
    response: Response = requests.get(f'{FIGSHARE_BASE_URL}collections/{collection_id}/articles?page_size={page_size}')
    os.makedirs(DESTINATION_FOLDER, exist_ok=True)

    for item in response.json():
        # Get the article download url
        article = requests.get(item['url']).json()
        files = article['files']
        if filter is not None:
            files = [file for file in files if filter in file['name']]

        for article_file in files:
            print(f'Downloading file: {article_file["name"]}')
            download_response: Response = requests.get(article_file['download_url'])
            if download_response.ok:
                print('OK')
                file_path = os.path.join(DESTINATION_FOLDER, article_file['name'])
                with open(file_path, 'wb') as file:
                    for chunk in download_response.iter_content(chunk_size=1024 * 8):
                        if chunk:
                            file.write(chunk)
                            file.flush()
                            os.fsync(file.fileno())
            else:
                print(f'Error downloading file {article_file["name"]}')


def delete_collection() -> None:
    files = glob.glob(f'{DESTINATION_FOLDER}/*')
    for file in files:
        os.remove(file)


def main():
    parser = argparse.ArgumentParser(description='Load the needed datasets')
    parser.add_argument(
        '-d',
        '--delete',
        required=False,
        action='store_true',
        help='Deletes all the datasets previously downloaded.',
    )
    parser.add_argument(
        '-f',
        '--filter',
        required=False,
        default=None,
        action='store',
        help='Filters the datasets to download by name.',
    )
    parser.add_argument(
        '-c',
        '--collection',
        default=COLLECTION_ID,
        action='store',
        required=False,
        help='Specifies a different collection to load.',
    )

    args = parser.parse_args()

    if args.delete:
        delete_collection()

    get_collection_articles(collection_id=args.collection, filter=args.filter)

if __name__ == '__main__':
    main()