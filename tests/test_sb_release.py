import tarfile

from release.build_layers import LAYER_MAX, build_app, build_weights


def test_app_layer_has_contract_paths(tmp_path):
    names = tarfile.open(build_app(tmp_path)).getnames()
    for want in ("app/app.py", "app/requirements.txt", "app/sourcebound/worker.py", "app/corpus", "app/output", "app/index"):
        assert want in names
    assert not any("__pycache__" in n for n in names)
    assert all(m.uid == 0 and m.mtime == 1_790_000_000 for m in tarfile.open(build_app(tmp_path)).getmembers())


def test_weights_split_into_bounded_layers(tmp_path, monkeypatch):
    import release.build_layers as bl
    monkeypatch.setattr(bl, "LAYER_MAX", 10)
    src = tmp_path / "m"
    (src / "sub").mkdir(parents=True)
    for i in range(3):
        (src / f"model-0000{i}.safetensors").write_bytes(b"x" * 6)
    (src / "sub" / "config.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    tars = bl.build_weights(out, src, "reader")
    assert len(tars) == 3
    names = [n for t in tars for n in tarfile.open(t).getnames()]
    assert "models/reader/sub/config.json" in names and "models/reader/model-00002.safetensors" in names
    assert LAYER_MAX == 4 * 2**30


def test_group_files_bounds_layers_and_isolates_big_shards():
    from release.build_layers import group_files
    items = [("a.safetensors", 3), ("b.safetensors", 3), ("c.safetensors", 12), ("config.json", 1)]
    assert group_files(items, limit=7) == [["a.safetensors", "b.safetensors"], ["c.safetensors"], ["config.json"]]
