"""Import the hyphenated audit script without re-running its main routine."""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("fetch_evidence", Path(__file__).with_name("fetch-evidence.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
capture, ROOT, TAG = module.capture, module.ROOT, module.TAG
