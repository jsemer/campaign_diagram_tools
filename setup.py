from setuptools import setup, find_packages

setup(
    name='campaign_diagram_tools',
    version='0.2',
    description='A package for visualizing kernel utilization using campaign diagrams',
    author='Joel S Emer, Toluwanimi Odemuyiwa',
    author_email='emer@csail.mit.edu, toluwa.odemuyiwa@gmail.com',
    url='https://github.com/jsemer/campaign_diagram_tools',
    packages=find_packages(),
    install_requires=[
        'altair',
        'pandas',
        'numpy',
        'matplotlib',
        'ruamel.yaml',
        'palettable',
        'deprecated',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
