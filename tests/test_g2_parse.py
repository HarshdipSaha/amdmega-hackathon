from gate.g2_hardware import parse_rocminfo


SAMPLE = """
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7940HS
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1103
  Marketing Name:          AMD Radeon 780M
  Device Type:             GPU
"""


def test_extracts_gpu_gfx_targets_only():
    assert parse_rocminfo(SAMPLE) == ["gfx1103"]


def test_empty_output_yields_no_targets():
    assert parse_rocminfo("") == []


def test_cpu_only_output_yields_no_targets():
    assert parse_rocminfo("Agent 1\n  Name:  AMD Ryzen\n  Device Type:  CPU\n") == []
