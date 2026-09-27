from gate.g1_transformers import supported_backend_names


def test_g1_transformers_exposes_three_explicit_sdpa_paths():
    assert supported_backend_names() == ("FLASH_ATTENTION", "MATH", "EFFICIENT_ATTENTION")
