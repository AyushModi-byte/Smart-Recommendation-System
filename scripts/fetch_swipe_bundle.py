"""Download and unpack the swipe app's data bundle (used by Render's build step).

The bundle (~130 MB: core.npz, places.db, weights.npz, meta.json) is too large for git. Build it with
training/16_export_swipe.py, zip it with   python scripts/fetch_swipe_bundle.py --pack   and upload the zip
somewhere public (a GitHub release asset works). Render then runs this script with SWIPE_BUNDLE_URL set.
"""
import argparse, os, shutil, sys, urllib.request, zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEST = os.path.join(ROOT, "bundle_swipe")
FILES = ["core.npz", "places.db", "weights.npz", "meta.json"]

ap = argparse.ArgumentParser()
ap.add_argument("--pack", action="store_true", help="zip ./bundle_swipe into ./bundle_swipe.zip instead of downloading")
ap.add_argument("--url", default=os.environ.get("SWIPE_BUNDLE_URL"))
a = ap.parse_args()

if a.pack:
    out = os.path.join(ROOT, "bundle_swipe.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(os.path.join(DEST, f), f)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    sys.exit(0)

if all(os.path.exists(os.path.join(DEST, f)) for f in FILES):
    print("bundle_swipe already present")
    sys.exit(0)
if not a.url:
    sys.exit("set SWIPE_BUNDLE_URL (or pass --url) to the bundle zip")
os.makedirs(DEST, exist_ok=True)
tmp = os.path.join(ROOT, "bundle_swipe.download.zip")
print("downloading", a.url, flush=True)
with urllib.request.urlopen(a.url, timeout=300) as r, open(tmp, "wb") as f:
    shutil.copyfileobj(r, f)
with zipfile.ZipFile(tmp) as z:
    for f in FILES:                      # extract only the expected names (no paths from the archive)
        with z.open(f) as src, open(os.path.join(DEST, f), "wb") as dst:
            shutil.copyfileobj(src, dst)
os.remove(tmp)
print("bundle_swipe ready:", ", ".join(f"{f} {os.path.getsize(os.path.join(DEST, f)) / 1e6:.0f} MB" for f in FILES))
