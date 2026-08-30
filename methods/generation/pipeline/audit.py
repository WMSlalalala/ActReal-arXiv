"""Generation-facing imports for the shared audit implementation."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from methods.shared.audit import (  # noqa: E402,F401
    assert_disjoint_user_splits,
    create_immutable_run_dir,
    reference_exclusion_audit,
    sha256_array,
    sha256_file,
    write_json_new,
)
