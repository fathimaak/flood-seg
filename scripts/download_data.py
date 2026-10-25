"""Download only the HandLabeled S2 chips, labels, and split CSVs from the public bucket."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.cloud import storage
from tqdm import tqdm

BUCKET = "sen1floods11"
ROOT = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"

# Remote prefix -> local folder
TARGETS = {
    "v1.1/splits/flood_handlabeled/": ROOT / "splits",
    "v1.1/data/flood_events/HandLabeled/S2Hand/": ROOT / "S2Hand",
    "v1.1/data/flood_events/HandLabeled/LabelHand/": ROOT / "LabelHand",
}


def main():
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET)

    jobs = []

    for prefix, local_dir in TARGETS.items():
        local_dir.mkdir(parents=True, exist_ok=True)

        for blob in client.list_blobs(BUCKET, prefix=prefix):
            if blob.name.endswith("/"):
                continue

            dest = local_dir / Path(blob.name).name

            if dest.exists() and dest.stat().st_size == blob.size:
                continue

            jobs.append((blob.name, dest))

    print(f"{len(jobs)} files to download")

    def fetch(job):
        name, dest = job
        bucket.blob(name).download_to_filename(str(dest))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(fetch, jobs), total=len(jobs)))

    print("Done ->", ROOT)


if __name__ == "__main__":
    main()
