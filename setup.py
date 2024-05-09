from setuptools import setup, find_packages

setup(
    author="Marios Kokmotos",
    description="",
    name="",
    version="0.1.0",
    packages=find_packages(include=["GPE_acceleration", "GPE_acceleration.*"]),
    install_requires=[
        'matplotlib>=3.7.1,<4',
        'numpy==1.23.5', 
        'pandas==1.5.3',
        'torch==2.1.0+cu118'
        ],
    python_requires='>=3.7',
)