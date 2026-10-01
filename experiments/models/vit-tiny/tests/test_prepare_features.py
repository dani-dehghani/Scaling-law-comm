from pathlib import Path

import prepare_features


def test_prepare_features_cli_passes_config_and_force(monkeypatch):
    captured = {}

    def fake_load_config(path):
        captured["config_path"] = Path(path)
        return {"loaded": True}

    def fake_prepare_feature_cache(cfg, project_root, force):
        captured["cfg"] = cfg
        captured["project_root"] = project_root
        captured["force"] = force

    monkeypatch.setattr(prepare_features, "load_config", fake_load_config)
    monkeypatch.setattr(
        prepare_features, "prepare_feature_cache", fake_prepare_feature_cache
    )
    monkeypatch.setattr(
        "sys.argv", ["prepare_features.py", "--config", "configs/example.yaml", "--force"]
    )

    prepare_features.main()

    expected_root = Path(prepare_features.__file__).resolve().parent
    assert captured["config_path"] == expected_root / "configs/example.yaml"
    assert captured["project_root"] == expected_root
    assert captured["cfg"] == {"loaded": True}
    assert captured["force"] is True
