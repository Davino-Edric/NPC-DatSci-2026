
import argparse
import shutil
import time
import zipfile
from pathlib import Path

COMPETITION = "schematics-2026"
OUT_DIR = Path("data") / COMPETITION
STAGING = OUT_DIR / "_tmp"


def list_all_files(api, prefix=""):
    """Ambil semua file kompetisi, menelusuri semua halaman."""
    files, token = [], None
    while True:
        res = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
        for f in res.files:
            size = getattr(f, "total_bytes", None) or getattr(f, "size", 0) or 0
            if f.name.startswith(prefix):
                files.append((f.name, int(size)))
        token = getattr(res, "next_page_token", None)
        print(f"  daftar file terkumpul: {len(files)}", end="\r")
        if not token:
            break
    print()
    return files


def download_one(api, name, size):
    target = OUT_DIR / name
    if target.exists() and (size == 0 or target.stat().st_size == size):
        return "skip"

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(STAGING, ignore_errors=True)
    STAGING.mkdir(parents=True)

    api.competition_download_file(
        COMPETITION, name, path=str(STAGING), force=True, quiet=True
    )

    base = Path(name).name
    found = next(STAGING.rglob(base), None)
    if found is None:  
        z = next(STAGING.rglob(base + ".zip"), None)
        if z is None:
            raise FileNotFoundError(f"Hasil unduhan tidak ditemukan: {name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(STAGING)
        found = next(STAGING.rglob(base), None)
        if found is None:
            raise FileNotFoundError(f"Gagal mengekstrak: {name}")

    shutil.move(str(found), str(target))
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="", help="hanya file dengan awalan ini, mis. train")
    parser.add_argument("--limit", type=int, default=0, help="batasi jumlah file (untuk tes)")
    parser.add_argument("--yes", action="store_true", help="lewati konfirmasi")
    args = parser.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Mengambil daftar file '{COMPETITION}' ...")
    files = list_all_files(api, args.prefix)
    if args.limit:
        files = files[: args.limit]

    total_gb = sum(s for _, s in files) / 1024**3
    print(f"Ditemukan {len(files)} file, total sekitar {total_gb:.2f} GB")
    if not files:
        return
    if not args.yes and input("Lanjut download? (y/n): ").strip().lower() != "y":
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, (name, size) in enumerate(files, 1):
        for attempt in range(1, 4):
            try:
                status = download_one(api, name, size)
                print(f"[{i}/{len(files)}] {status}: {name}")
                break
            except Exception as e:
                print(f"[{i}/{len(files)}] gagal (percobaan {attempt}/3): {name} -> {e}")
                time.sleep(2 * attempt)
        else:
            failed.append(name)

    shutil.rmtree(STAGING, ignore_errors=True)

    print(f"\nSelesai. Berhasil/dilewati: {len(files) - len(failed)}, gagal: {len(failed)}")
    if failed:
        (OUT_DIR / "failed_files.txt").write_text("\n".join(failed), encoding="utf-8")
        print(f"Daftar file gagal disimpan di {OUT_DIR / 'failed_files.txt'}; jalankan ulang untuk mencoba lagi.")


if __name__ == "__main__":
    main()