#!/usr/bin/env python3
from pathlib import Path
import re,sys
ROOT=Path(__file__).resolve().parents[1]
img={".png",".jpg",".jpeg",".tif",".tiff",".gif",".bmp",".webp",".svg"}
raw={".dbc",".dbf",".pbf",".rda",".rdata",".sav",".sas7bdat"}
bad=[]
for p in ROOT.rglob("*"):
 if not p.is_file() or ".git" in p.parts or p.resolve() == Path(__file__).resolve(): continue
 rel=p.relative_to(ROOT)
 if p.suffix.lower() in img: bad.append(f"IMAGE_BINARY\t{rel}")
 if p.suffix.lower() in raw: bad.append(f"RAW_DATA_EXT\t{rel}")
 if p.stat().st_size>25*1024*1024: bad.append(f"LARGE_GT25MB\t{rel}")
 if p.suffix.lower() in {".md",".txt",".csv",".json",".yaml",".yml",".py",".toml",".cff"}:
  t=p.read_text(errors="ignore")
  if re.search(r"/Users/[^/\s]+/|/Volumes/[^/\s]+/",t): bad.append(f"PRIVATE_PATH\t{rel}")
  if re.search(r"sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}",t): bad.append(f"TOKEN_PATTERN\t{rel}")
if bad:
 print("\n".join(bad));sys.exit(1)
print("PASS: no prohibited images, raw-data extensions, >25MB files, obvious private paths, or token patterns.")
