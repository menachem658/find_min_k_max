from setuptools import setup, find_packages

setup(
    name='find_min_k_max',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy'
    ],
    author='menachem sokolik',
    author_email='menachemsokolik@gmail.com',
    description='A package for statistical computations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
