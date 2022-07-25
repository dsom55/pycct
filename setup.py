import io
import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

INSTALL_REQUIRES = [
    'numpy>=1.21',
    'scipy>=1.6',
    'matplotlib>=3.3',
    'pandas>=1.2'
]

setuptools.setup(
    name='pycct',
    version='0.0.1',
    author='David Sommer',
    author_email='dsommer55@gmail.com',
    description="Defect thermodynamics in semiconducting and insulating compounds.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    packages=setuptools.find_packages(include=['pycct, pycct.*']),
    install_requires=INSTALL_REQUIRES
    )
