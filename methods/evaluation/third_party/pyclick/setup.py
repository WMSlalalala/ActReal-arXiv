import setuptools
from pathlib import Path

readme = Path(__file__).with_name("README.md")
long_description = (
    readme.read_text(encoding="utf-8")
    if readme.is_file()
    else "Human mouse movement simulation with Python."
)

setuptools.setup(
    name='pyclick',
    version='0.0.1',
    author='patrikoss',
    description='Human mouse movement simulation with python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/patrikoss/pyclick",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.14.2",
        "PyAutoGUI>=0.9.36",
        "PyTweening>=1.0.3",
        "python3-xlib>=0.15",
        "xlib>=0.21",
    ],
)
