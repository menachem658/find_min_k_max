from setuptools import setup, find_packages

setup(
    name="non_frequent_symbols",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
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
