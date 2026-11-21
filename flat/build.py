"""Inline the classroom picture, cat sprites and desk cut-outs into one HTML page.

Usage: python3 flat/build.py   ->  flat/dist/study-office.html
"""
import base64, json, pathlib

ROOT = pathlib.Path(__file__).parent
A = ROOT / "assets"
assets = {}
for f in sorted(A.glob("*.webp")):
    assets[f.stem] = "data:image/webp;base64," + base64.b64encode(f.read_bytes()).decode()
occ = json.loads((A / "occ.json").read_text())
html = (ROOT / "src" / "index.html").read_text(encoding="utf-8")
html = html.replace("/*ASSETS*/{}", json.dumps(assets)).replace("/*OCC*/[]", json.dumps(occ))
out = ROOT / "dist" / "study-office.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
print(out, f"{out.stat().st_size / 1024:.0f} KB")
