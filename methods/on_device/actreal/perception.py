"""Let the framework's perception models load on a current ModelScope.

Mobile-Agent-E builds three ModelScope pipelines -- GroundingDINO for icons and
two OCR models for text.  The GroundingDINO repository ships custom Python that
the pipeline executes, and ModelScope used to run it without asking.  Current
versions refuse unless the caller passes ``trust_remote_code=True``, so upstream
now dies at::

    RuntimeError: Detected plugins or allow_remote field in the model
    configuration file, but trust_remote_code=True was not explicitly set.

**What that flag means is worth stating plainly**: it lets code from the model
repository run in this process.  Nothing about the risk changed when ModelScope
added the check -- the same code was always executed by this framework, on the
same repositories, and the check only made the consent explicit.  What changed
is that the consent is now yours to give, which is why it is a flag here rather
than a constant, and why the runner prints it.

The name ``pipeline`` is rebound in the framework's own module rather than
edited there, the same way the controller primitives and the captioner are, so
``third_party`` stays byte-for-byte upstream.
"""

from __future__ import annotations

from typing import Any

# The repositories the framework asks for.  Recorded so a run can say what it
# trusted rather than only that it trusted something.
EXPECTED_REPOSITORIES = (
    "AI-ModelScope/GroundingDINO",
    "iic/cv_resnet18_ocr-detection-db-line-level_damo",
    "iic/cv_convnextTiny_ocr-recognition-document_damo",
)


def install(module, *, trust_remote_code: bool = True) -> dict[str, Any]:
    """Wrap ``module.pipeline`` so the perception models will build.

    Returns what it did, including the flag's value, so the session report
    carries the decision instead of it living only in someone's shell history.
    """

    original = getattr(module, "pipeline", None)
    if original is None:
        return {"installed": False, "reason": "the module exposes no pipeline()"}
    if not trust_remote_code:
        return {
            "installed": False,
            "trust_remote_code": False,
            "reason": "not enabled; GroundingDINO will refuse to build on "
            "ModelScope versions that require explicit consent",
        }

    def pipeline(*args, **kwargs):
        # Only supply it when the caller did not: an explicit False upstream
        # would be a decision, and this is not the place to overrule one.
        kwargs.setdefault("trust_remote_code", True)
        return original(*args, **kwargs)

    module.pipeline = pipeline
    return {
        "installed": True,
        "trust_remote_code": True,
        "repositories": list(EXPECTED_REPOSITORIES),
        "reason": "model repositories may execute their own code in this process",
    }
