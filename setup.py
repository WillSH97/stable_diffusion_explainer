from setuptools import setup, find_packages

setup(
    name="SD_utils",
    version="0.0.1",
    author="QUT GenAI Lab",
    author_email="william.he@qut.edu.au",
    description="SD Utils for SD v1.4 explainer",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "accelerate",
        "PIL",
        "requests",
        "io",
        "numpy",
        "matplotlib",
        "
    ],
)