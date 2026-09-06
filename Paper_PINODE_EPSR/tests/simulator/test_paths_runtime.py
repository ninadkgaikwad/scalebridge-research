import tempfile
import unittest
from pathlib import Path

from pinode_epsr.simulator.paths import EPSRProjectLayout
from pinode_epsr.simulator.runtime_idf import (
    prepare_300s_runtime_idf,
    sha256_file,
)


class PathRuntimeTests(unittest.TestCase):
    def test_root_derivation(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "BuildingModelingProject_Condensed"
            repo = (
                base
                / "NewOrg"
                / "scalebridge-research"
                / "Paper_PINODE_EPSR"
            )
            repo.mkdir(parents=True)

            layout = EPSRProjectLayout.from_repo_root(repo)

            self.assertEqual(
                layout.data_root,
                (
                    base
                    / "Data"
                    / "ScaleBridge"
                    / "Paper_PINODE_EPSR"
                ).resolve(),
            )

    def test_runtime_copy_only(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "model.idf"
            dst = td / "run" / "model.idf"

            src.write_text(
                "Version,24.1;\nTimestep,6;\n",
                encoding="utf-8",
            )
            h = sha256_file(src)

            prepare_300s_runtime_idf(
                authoritative_idf=src,
                runtime_idf=dst,
                expected_source_sha256=h,
            )

            self.assertEqual(sha256_file(src), h)
            self.assertIn(
                "Timestep,12;",
                dst.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
