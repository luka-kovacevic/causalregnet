from distutils.core import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name='causalregnet',
    version='0.1',
    python_requires=">=3.8",
    packages=find_packages(),
    package_data={
        "": ["*.txt"]
    },
    author='see README.txt',
    url="github.com/structure_learning",
    author_email='luka.kovacevic@mrc-bsu.cam.ac.uk',
    license="Apache-2.0",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires #,
    # entry_points={
    #     'console_scripts': [
    #         'causalbench_run=causalscbench.apps.main_app:main',
    #     ],
    # },
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: Apache Software License",
    #     "Operating System :: OS Independent",
    # ]
)