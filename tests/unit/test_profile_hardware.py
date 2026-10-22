import hashlib
from scripts.profile_hardware import build_inventory


def test_inventory_records_available_memory_and_lock_identity(tmp_path):
    # Catches missing available RAM, lost lock identity, and falsely asserted matches.
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"fixture lock\n")
    inventory = build_inventory(tmp_path)
    assert inventory["schema_version"] == 1
    assert 0 < inventory["memory"]["available_bytes"] <= inventory["memory"]["total_bytes"]
    assert inventory["cpu"]["logical_cores"] >= 1
    assert inventory["reference"]["logical_cores"] == 12
    assert inventory["reference"]["installed_gib"] == 15.33
    assert inventory["locks"]["uv.lock"]["sha256"] == hashlib.sha256(b"fixture lock\n").hexdigest()
    assert inventory["locks"]["package-lock.json"] is None
    assert inventory["power"]["active_scheme"]["status"] in {"observed", "unavailable"}
    assert inventory["resource_settings_status"] == "provisional; not benchmarked or enforced"
    assert inventory["reference_comparison"]["status"] in {"match", "mismatch", "incomplete"}
    assert not (tmp_path / "inventory").exists()
