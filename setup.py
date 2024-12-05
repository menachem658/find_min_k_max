from setuptools import setup, find_packages

setup(
    name='find_min_k_max',
    version='0.1.0',
    description='A Python package to calculate the minimum k_max for statistical bounds.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/username/find_min_k_max',
    author='menachem sokolik',
    author_email='menachemsokolik@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
