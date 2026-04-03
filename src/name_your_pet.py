import csv
import hashlib
import socket
from pathlib import Path

# Warn if not running on the remote server
if "heinz-90803" not in socket.gethostname():
    print("Warning: this script is meant to be run on the remote server.")

repo_root = Path(__file__).parent.parent
data_file = repo_root / "data" / "pet_data.csv"
names_file = repo_root / "data" / "pet_names.csv"

with open(data_file) as f:
    row = next(csv.DictReader(f))
    name = row["name"].strip()
    animal = row["animal"].strip()
    color = row["color"].strip()

with open(names_file) as f:
    pet_names = [row["name"] for row in csv.DictReader(f) if row["name"].strip()]

seed = f"{name.lower()}|{animal.lower()}|{color.lower()}"
digest = hashlib.sha256(seed.encode()).digest()
name_idx = int.from_bytes(digest[:8], "big") % len(pet_names)

print(f"Your {color} {animal}'s name is: {pet_names[name_idx]}")
