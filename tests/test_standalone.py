"""Contracts for a checkout with no adjacent EC66 projects or cached paths."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap import ROOT, RUNTIME


class StandaloneTests(unittest.TestCase):
    def test_scene_assets_and_snapshot_are_inside_checkout(self):
        for name in ("scene_config.json", "scene_config_particles.json"):
            config = json.loads((RUNTIME / name).read_text())
            for key in ("urdf", "shovel_mesh", "particle_positions_file"):
                if key in config:
                    path = (RUNTIME / config[key]).resolve()
                    self.assertTrue(path.is_relative_to(ROOT))
                    self.assertTrue(path.is_file())
            urdf = RUNTIME / config["urdf"]
            for mesh in ET.parse(urdf).findall(".//mesh"):
                path = (urdf.parent / mesh.get("filename")).resolve()
                self.assertTrue(path.is_relative_to(ROOT))
                self.assertTrue(path.is_file())
        particles = json.loads((RUNTIME / "scene_config_particles.json").read_text())
        self.assertFalse(particles["joint_trajectory_enabled"])
        self.assertNotIn("joint_trajectory_file", particles)
        snapshot = json.loads((RUNTIME / particles["particle_positions_file"]).read_text())
        self.assertEqual(len(snapshot["positions_m"]), particles["fixed_particle_count"])

    def test_runtime_imports_are_bundled(self):
        for name in ("kinematics", "trajectory_planning", "emcp_phase1_planning",
                     "full_state_controller", "pose_controller", "motion_join",
                     "retention_physics", "particle_distribution", "record_particle_replay",
                     "release_reference_viewer", "particle_scene", "my_scene", "granules",
                     "distribution_overlay"):
            module = importlib.import_module(name)
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(RUNTIME), name)

    def test_release_build_after_relocation_without_history(self):
        with tempfile.TemporaryDirectory(prefix="ec66 standalone ") as folder:
            checkout = Path(folder) / "checkout"
            checkout.mkdir()
            for path in ROOT.glob("*.py"):
                shutil.copy2(path, checkout / path.name)
            for name in ("runtime", "schemes"):
                shutil.copytree(ROOT / name, checkout / name,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
            result = subprocess.run([sys.executable, "-s", str(checkout / "cli.py"), "build",
                                     "--scheme", str(checkout / "schemes/01_default.json")],
                                    cwd=folder, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            runs = list((checkout / "outputs").glob("*/status.json"))
            self.assertEqual(len(runs), 1)
            self.assertEqual(json.loads(runs[0].read_text())["state"], "complete")
            sources = json.loads((runs[0].parent / "provenance.json").read_text())["files_sha256"]
            self.assertIn("runtime/assets/ec66_description.urdf", sources)
            self.assertTrue(all(not Path(p).is_absolute() for p in sources))


if __name__ == "__main__":
    unittest.main()
