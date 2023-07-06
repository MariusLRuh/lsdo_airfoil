from setuptools import setup, find_packages

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lsdo_airfoil',
    version=get_version('lsdo_airfoil/__init__.py'),
    author='Marius Ruh',
    author_email='mruh@ucsd.edu',
    license='LGPLv3+',
    keywords='lsdo airfoil repository package',
    url='http://github.com/LSDOlab/lsdo_airfoil',
    # download_url='http://pypi.python.org/pypi/lsdo_project_template',
    description='A machine learning model for airfoil aerodynamic prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.7',
    platforms=['any'],
    install_requires=[
        'numpy',
        'pytest',
        'myst-nb',
        'sphinx',
        'sphinx_rtd_theme',
        'sphinx-copybutton',
        'sphinx-autoapi',
        'numpydoc',
        'gitpython',
        # 'sphinxcontrib-collections @ git+https://github.com/anugrahjo/sphinx-collections.git', # 'sphinx-collections',
        'sphinxcontrib-bibtex',
        'setuptools',
        'wheel',
        'twine',
        'csdl @ git+https://github.com/LSDOlab/csdl.git',
        'scipy',
        'python_csdl_backend @ git+https://github.com/LSDOlab/python_csdl_backend.git',
        'torch',
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Natural Language :: English',
    ],
)
