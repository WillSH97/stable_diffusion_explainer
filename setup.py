from setuptools import setup, find_packages

setup(
    name="SDutils",
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
        "pillow",
        "requests",
        "numpy",
        "matplotlib",
    ],
)
