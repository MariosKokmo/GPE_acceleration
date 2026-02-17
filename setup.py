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
    package_dir={"": "src"},
    packages=find_packages(where='src', 
                           include=["src*"],
                           ),
    entry_points={
        "console_scripts": [
            "baqs=src.cli.baqs:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        'matplotlib>=3.7.1,<4',
        'numpy>=1.23.5,<1.3', 
        'pandas>=1.5.3,<1.6',
        'torch==2.1.0+cu118'
        ],
    python_requires='>=3.7',
)