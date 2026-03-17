#!/usr/bin/env python

# Use README.md as description:
try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except: long_description = ''
        
# call setup:
from setuptools import setup
setup(
    name='rag-exp',
    version='0.1',
    description='A Python package for explaining RAG pipelines.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/k-randl/Interpretable_RAG',
    author='Korbinian Randl & Guido Rocchietti',
    author_email='k.randl@web.de',
    packages=['Interpretable_RAG'],
    package_dir = {'': 'src'},
    install_requires=[
        'accelerate',
        'ipython',
        'matplotlib',
        'nltk',
        'numpy',
        'pandas',
        'sentence-transformers',
        'torch',
        'tqdm',
        'transformers',
    ],
)
