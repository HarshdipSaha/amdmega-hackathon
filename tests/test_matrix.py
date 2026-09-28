import pytest

from silentpath.matrix import expand, load_matrix

YAML = """
model: Qwen/Qwen2.5-1.5B-Instruct
seed: 0
max_tokens: 64
backends: [ROCM_ATTN, TRITON_ATTN]
workloads:
  - id: invoice_01
    prompt: "What is the total?"
  - id: invoice_02
    prompt: "What is the tax?"
"""


def test_expand_produces_backend_by_workload_cells():
    cells = expand(load_matrix(YAML))
    assert len(cells) == 4
    assert {(c.backend, c.workload_id) for c in cells} == {
        ("ROCM_ATTN", "invoice_01"), ("ROCM_ATTN", "invoice_02"),
        ("TRITON_ATTN", "invoice_01"), ("TRITON_ATTN", "invoice_02")}


def test_cell_config_carries_model_and_seed():
    c = expand(load_matrix(YAML))[0]
    assert c.config["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert c.config["seed"] == 0


def test_cell_config_fingerprints_prompt_without_storing_its_text():
    cell = expand(load_matrix(YAML))[0]
    assert "prompt" not in cell.config
    assert len(cell.config["prompt_sha256"]) == 64


def test_changed_prompt_changes_cache_identity():
    first = expand(load_matrix(YAML))[0]
    changed = expand(load_matrix(YAML.replace("What is the total?", "What is the new total?")))[0]
    assert first.config["prompt_sha256"] != changed.config["prompt_sha256"]


def test_duplicate_workload_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        expand(load_matrix(YAML + "\n  - id: invoice_01\n    prompt: 'dup'\n"))


def test_empty_backends_rejected():
    with pytest.raises(ValueError, match="backend"):
        expand(load_matrix("model: m\nbackends: []\nworkloads: [{id: a, prompt: p}]"))
