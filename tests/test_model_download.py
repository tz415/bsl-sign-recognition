"""Tests for model-file integrity helpers."""

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bsl_sign_recognition.model_download import sha256_file


class ModelHashTests(unittest.TestCase):
    def test_sha256_file_matches_hashlib(self) -> None:
        content = b"BSL hand landmark model test"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.task"
            path.write_bytes(content)
            self.assertEqual(
                sha256_file(path),
                hashlib.sha256(content).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
