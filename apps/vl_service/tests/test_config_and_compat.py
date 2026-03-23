from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


class SettingsDefaultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_keys = (
            "VL_SERVICE_PIPELINE_VERSION",
            "VL_SERVICE_LAYOUT_DETECTION_MODEL_NAME",
            "VL_SERVICE_LAYOUT_DETECTION_MODEL_DIR",
            "VL_SERVICE_VL_REC_MODEL_NAME",
            "VL_SERVICE_VL_REC_MODEL_DIR",
            "VL_SERVICE_USE_SEAL_RECOGNITION",
        )
        self._env_backup = {key: os.environ.get(key) for key in self._env_keys}
        for key in self._env_keys:
            os.environ.pop(key, None)

        from app.core.config import get_settings

        get_settings.cache_clear()

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        from app.core.config import get_settings

        get_settings.cache_clear()

    def test_v1_auto_detects_local_vendor_bundle(self) -> None:
        os.environ["VL_SERVICE_PIPELINE_VERSION"] = "v1"

        from app.core.config import (
            DEFAULT_V1_LAYOUT_MODEL_DIR,
            DEFAULT_V1_VENDOR_ROOT,
            get_settings,
        )

        get_settings.cache_clear()
        settings = get_settings()

        self.assertEqual(settings.pipeline_version, "v1")
        self.assertEqual(settings.vl_rec_model_dir, DEFAULT_V1_VENDOR_ROOT)
        self.assertEqual(
            settings.layout_detection_model_dir,
            DEFAULT_V1_LAYOUT_MODEL_DIR,
        )
        self.assertEqual(settings.vl_rec_model_name, "PaddleOCR-VL-0.9B")
        self.assertEqual(settings.layout_detection_model_name, "PP-DocLayoutV2")

    def test_v1_5_defaults_disable_seal_recognition(self) -> None:
        os.environ["VL_SERVICE_PIPELINE_VERSION"] = "v1.5"

        from app.core.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        self.assertEqual(settings.pipeline_version, "v1.5")
        self.assertFalse(settings.use_seal_recognition)


class PaddleXCompatShimTests(unittest.TestCase):
    def test_install_shim_registers_missing_official_models_module(self) -> None:
        module_name = "paddlex.inference.utils.official_models"
        original_module = sys.modules.pop(module_name, None)
        try:
            from app.inference.paddle_vl_backend import (
                _install_paddlex_official_models_shim,
            )

            _install_paddlex_official_models_shim()
            shim_module = sys.modules[module_name]

            self.assertTrue(hasattr(shim_module, "official_models"))
            self.assertIsNotNone(shim_module.official_models)
        finally:
            if original_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original_module


if __name__ == "__main__":
    unittest.main()