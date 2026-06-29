import os
import unittest
from unittest.mock import patch

from app.config import get_runtime_settings


class RuntimeSettingsTests(unittest.TestCase):
    def test_runtime_settings_use_project_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_runtime_settings()

            self.assertEqual(settings["port"], 8001)
            self.assertTrue(str(settings["chroma_db_path"]).endswith("chroma_db"))

    def test_runtime_settings_respect_environment_overrides(self):
        with patch.dict(os.environ, {"PORT": "9000", "CHROMA_DB_PATH": "/tmp/custom-chroma"}, clear=True):
            settings = get_runtime_settings()

            self.assertEqual(settings["port"], 9000)
            self.assertEqual(settings["chroma_db_path"], "/tmp/custom-chroma")


if __name__ == "__main__":
    unittest.main()
