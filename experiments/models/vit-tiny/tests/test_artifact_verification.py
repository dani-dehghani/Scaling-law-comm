import csv
import json
from pathlib import Path
import shutil
import pytest

from cli.verify_artifacts import check_hpo, check_sweep


def _sweep(folder):
    (folder / "config_snapshot.json").write_text(json.dumps({"data": {"dataset_sizes": [200], "seeds": [0, 1]}, "probe": {}}))
    rows = [{"D": 200, "seed": 0, "test_loss": 1.0, "test_accuracy": 0.5},
            {"D": 200, "seed": 1, "test_loss": 1.0, "test_accuracy": 0.5}]
    for name, values in [("summary.csv", rows), ("aggregated_by_D.csv", [{"D": 200, "test_loss_mean": 1.0,
            "test_loss_std": 0.0, "test_accuracy_mean": 0.5, "test_accuracy_std": 0.0, "n_seeds": 2}])]:
        with (folder / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=values[0]); writer.writeheader(); writer.writerows(values)
    (folder / "scaling_fit.json").write_text(json.dumps({"L_inf": 0.5, "b": 1.0, "beta": 0.5}))


def test_verifier_detects_duplicate_identity(tmp_path):
    _sweep(tmp_path)
    path = tmp_path / "summary.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([*lines, lines[-1]]) + "\n")
    with pytest.raises(ValueError, match="duplicate D/seed"):
        check_sweep(tmp_path)


def test_verifier_checks_aggregate_arithmetic(tmp_path):
    _sweep(tmp_path)
    assert check_sweep(tmp_path)["runs"] == 2
    path = tmp_path / "aggregated_by_D.csv"
    path.write_text(path.read_text().replace("200,1.0", "200,9.0"))
    with pytest.raises(ValueError, match="incorrect test_loss_mean"):
        check_sweep(tmp_path)


def test_verifier_rejects_hpo_selection_that_disagrees_with_trials(tmp_path):
    source = Path(__file__).resolve().parents[1] / "artifacts" / "places365_boundary_hpo"
    for name in ("config_snapshot.json", "hpo_trials.csv", "hpo_aggregated.csv", "selected_global.json", "completion.json"):
        shutil.copyfile(source / name, tmp_path / name)
    assert check_hpo(tmp_path)["trials"] == 1920
    path = tmp_path / "selected_global.json"
    selection = json.loads(path.read_text())
    selection["learning_rate"] = 0.00003
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="wrong global selection"):
        check_hpo(tmp_path)
