"""Protect reproducible datasets and the boundary of the public model bundle."""
import contextlib
import io
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from scripts.package_model import REQUIRED, package

ROOT = Path(__file__).resolve().parent


class PublicationTest(unittest.TestCase):
    def test_public_entry_points_import_without_local_diagnostics(self):
        import runpy
        import sys
        from unittest.mock import patch
        for name in ("agent.py", "verify_run.py", "run_comparison.py", "render_comparison.py", "serve_doom_laya.py", "training/finetune.py"):
            with self.subTest(name=name), patch.object(sys, "argv", [name, "--help"]), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(ROOT / name), run_name="__main__")
                self.assertEqual(result.exception.code, 0)

    def test_frozen_training_data_hashes_and_holdout(self):
        manifest = json.loads((ROOT / 'training/datasets.json').read_text())
        sources = {}
        for name, expected in manifest.items():
            content = (ROOT / name).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected['sha256'], name)
            rows = json.loads(content)
            self.assertEqual(len(rows), expected['questions'])
            sources[name] = {r['source_run'] for r in rows}
            for row in rows:
                self.assertIn(row['label'], row['question']['criteria'])
                self.assertNotIn('48', row['source_run'].rsplit('-', 1)[-1])
        for base in ('training', 'training/v3'):
            self.assertFalse(sources[base + '/train.json'] & sources[base + '/validation.json'])

    def test_bundle_excludes_secrets_and_scrubs_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'checkpoint', Path(tmp) / 'model'
            for name in REQUIRED:
                p = source / name
                p.parent.mkdir(parents=True, exist_ok=True)
                if name.endswith('.json'):
                    private = '/' + 'Users' + '/example/private/checkpoint'
                    p.write_text(json.dumps({'base_checkpoint': private}))
                else:
                    p.write_bytes(b'tiny test weights')
            (source / '.env').write_text('PRIVATE=must not escape')
            (source / 'unrelated.log').write_text('not public')
            package(source, output, archive=True)
            self.assertFalse((output / '.env').exists())
            self.assertFalse((output / 'unrelated.log').exists())
            config = json.loads((output / 'rl_agent_config.json').read_text())
            self.assertEqual(config['base_checkpoint'], 'checkpoint')
            self.assertEqual((output / 'model.safetensors').read_bytes(), (source / 'model.safetensors').read_bytes())
            for line in (output / 'SHA256SUMS').read_text().splitlines():
                digest, name = line.split('  ', 1)
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            with tarfile.open(output.with_suffix('.tar')) as archive:
                self.assertTrue(all(m.name.startswith('model/') and m.isfile() for m in archive.getmembers()))
                self.assertNotIn('model/.env', archive.getnames())
            with self.assertRaises(FileExistsError):
                package(source, output)


if __name__ == '__main__':
    unittest.main()
