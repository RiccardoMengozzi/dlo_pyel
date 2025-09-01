from setuptools import setup, find_packages

setup(
    name="pyel_model",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "wandb",
        "matplotlib",
        "wandb",
        "tqdm",
        "shutils",
        "numpy",
        "scipy",
    ],
)
