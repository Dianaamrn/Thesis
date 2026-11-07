"""Step 0: download the MONDO ontology once (for channel C3, disease categories).

The release is pinned by keeping this one file for the whole experiment; its version is read from
the file's data-version line and saved with every result.

  python phase2/step0_get_mondo.py            # downloads to data/mondo.obo (~50 MB)
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings  # noqa: E402
import lib_data  # noqa: E402


def main():
    path = Path(settings.MONDO['file'])
    if path.exists():
        print(f'{path} already exists (delete it to download another release)')
    else:
        print(f'downloading {settings.MONDO["url"]} -> {path}')
        tmp = path.with_suffix('.part')
        urllib.request.urlretrieve(settings.MONDO['url'], tmp)
        tmp.replace(path)
    parents, names, version, obsolete = lib_data.read_mondo(path)
    root = settings.MONDO['root'] if settings.MONDO['root'] in parents else settings.MONDO['fallback_root']
    level = {root}
    for _ in range(settings.MONDO['depth']):
        level = {t for t, ps in parents.items() if t not in obsolete and any(p in level for p in ps)}
    print(f'MONDO release {version}: {len(parents):,} terms; {len(level)} categories '
          f'{settings.MONDO["depth"]} levels below {root} ({names.get(root)})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
