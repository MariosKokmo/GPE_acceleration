from setuptools import setup, find_packages
import os, codecs, re

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts),'r') as fp:
        return fp.read()

def find_version(*file_paths):
    """
    Find the version of the software using regex
    """
    version_file = read(*file_paths)
    reg = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}")
    match = reg.search(version_file)
    if match:
        return match.group(0)
    raise RuntimeError("Unable to find version string")

setup(
    author="Marios Kokmotos",
    description="",
    name="baqs",
    version=find_version('src','__version__.py'),
    # Install ``src`` as a top-level importable package so that the
    # ``from src....`` imports used throughout the codebase resolve when the
    # distribution is pip-installed (with no source tree present).
    packages=find_packages(include=["src", "src.*"]),
    entry_points={
        "console_scripts": [
            "baqs=src.cli.baqs:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        'matplotlib>=3.8,<4',
        'numpy>=1.26,<2.0',
        'pandas>=2.1,<2.3',
        'opencv-python>=4.9',
        'torch>=2.2,<2.5'
        ],
    python_requires='>=3.9,<3.13',
)