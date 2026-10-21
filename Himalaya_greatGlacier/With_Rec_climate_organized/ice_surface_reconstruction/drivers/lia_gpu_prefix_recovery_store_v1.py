"""Read-only inherited results/checkpoints with one new output writer."""

from pathlib import Path
import json

from lia_gpu_four_group_search_v2 import file_hash
from lia_gpu_matrix_io_v2 import Store


class RecoveryStore(Store):
    def __init__(self, root, source, inventory):
        super().__init__(root)
        self.source = Store(source)
        self.inventory = inventory

    def entry(self, kind, key):
        if kind != 'spin' or key not in self.inventory:
            raise ValueError('candidate outside registered group-one inventory')
        return self.inventory[key]

    def check_files(self, entry):
        for item in entry.get('files', []):
            path = (self.source.root / item['path']).resolve()
            if not path.is_relative_to(self.source.root.resolve()):
                raise ValueError('inherited path escapes source')
            if file_hash(path) != item['sha256']:
                raise ValueError('inherited evidence changed: ' + item['path'])

    def read(self, kind, key):
        entry = self.entry(kind, key)
        local = super().read(kind, key)
        if entry['mode'] == 'terminal':
            if local is not None:
                raise ValueError('completed candidate was duplicated')
            self.check_files(entry)
            return entry['result']
        return local

    def arrays(self, kind, key):
        entry = self.entry(kind, key)
        if entry['mode'] == 'terminal':
            self.check_files(entry)
            return self.source.arrays(kind, key)
        return super().arrays(kind, key)

    def latest(self, kind, key):
        entry = self.entry(kind, key)
        local = super().latest(kind, key)
        if local is not None:
            return local
        if entry['mode'] == 'checkpoint':
            self.check_files(entry)
            saved = self.source.latest(kind, key)
            if saved is None or saved[0]['year'] != entry['year']:
                raise ValueError('inherited checkpoint changed')
            return saved
        return None


def inventory_source(root, search, manifest):
    import numpy as np
    root = Path(root)
    store = Store(root)
    nx = {r['system']: r['nx'] for r in manifest['systems']}
    inventory = {}

    def files(directory, result, manifest_name='result.json'):
        paths = [directory / manifest_name] + [directory / a['name'] for a in result.get('artifacts', [])]
        return [dict(path=str(p.relative_to(root)).replace('\\', '/'), sha256=file_hash(p)) for p in paths]

    for c in search.candidates(1, []):
        key = search.spin_key(c)
        row = store.read('spin', key)
        quarantine = root / 'quarantine' / ('spin-' + key[:16] + '.json')
        if row is not None:
            if row.get('candidate') != c.__dict__:
                raise ValueError('terminal candidate identity mismatch')
            inventory[key] = dict(mode='terminal', result=row, files=files(store.directory('spin', key), row))
        elif quarantine.exists():
            row = json.loads(quarantine.read_text())
            if row.get('key') != key or row.get('status') != 'technical-failure':
                raise ValueError('quarantine identity mismatch')
            inventory[key] = dict(mode='terminal', result=row, files=[dict(path=str(quarantine.relative_to(root)).replace('\\', '/'), sha256=file_hash(quarantine))])
        else:
            cp = store.latest('spin', key)
            if cp is None:
                inventory[key] = dict(mode='new', files=[])
                continue
            meta, arrays = cp
            year = meta['year']
            if year not in (330, 630, 930, 1230):
                raise ValueError('unexpected source checkpoint phase')
            if arrays['thick'].shape != (nx[c.system],) or arrays['totals'].shape != (2,):
                raise ValueError('source checkpoint shape mismatch')
            years = arrays['years']
            if not np.array_equal(years, np.arange(0, year + 1, 30)) or arrays['history'].shape != (len(years), 5):
                raise ValueError('source checkpoint history mismatch')
            if not all(np.isfinite(arrays[k]).all() for k in arrays) or np.any(arrays['thick'] < 0):
                raise ValueError('source checkpoint invalid state')
            counters = meta['counters']
            if counters['accepted_steps'] <= 0 or counters['rejected_steps'] < 0 or not np.isfinite(counters['minimum_actual_timestep_seconds']):
                raise ValueError('source checkpoint counters invalid')
            directory = root / 'resume' / 'spin' / key[:16] / f'y{year:04d}' / 'checkpoint' / key[:16]
            inventory[key] = dict(mode='checkpoint', year=year, counters=counters, files=files(directory, meta))
    if len(inventory) != 4011:
        raise ValueError('incomplete population inventory')
    return inventory
