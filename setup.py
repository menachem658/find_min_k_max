from setuptools import setup, find_packages

setup(
    name="find_min_k_max",
    version="0.1.0",
    author="menachem sokolik",
    author_email="menachemsokolik@gmail.com",
    description="A Python package for statistical analysis of non-frequent symbols.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
