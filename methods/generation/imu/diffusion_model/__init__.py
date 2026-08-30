"""ActReal action-conditioned IMU diffusion model package.

This package is intentionally self-contained.  It does not import detector
training/evaluation code; adversarial training uses the lightweight critics in
``diffusion_model.critics`` only.
"""

from .utils import GENERATOR_VERSION

__version__ = GENERATOR_VERSION
