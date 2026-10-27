from pathlib import Path
import gzip,hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'UPLOAD_MANIFEST.json').read_text())
for row in manifest['files']:
    stored=root/row['stored']
    data=gzip.decompress(stored.read_bytes()) if row['gzip'] else stored.read_bytes()
    if hashlib.sha256(data).hexdigest()!=row['sha256']:
        raise RuntimeError('Checksum mismatch: '+row['path'])
    if row['gzip']:
        target=root/row['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
print('All snapshot files verified; original large files restored.')
