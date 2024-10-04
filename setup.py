from setuptools import setup, find_packages

setup(
    name='campaign_diagram_tools',
    version='0.1',
    description='A package for visualizing kernel utilization using campaign diagrams',
    author='Joel S Emer',
    author_email='emer@csail.mit.edu',
    url='https://github.com/jsemer/campaign_diagram_tools',
    packages=find_packages(),
    install_requires=[
        'matplotlib',  # Dependency for plotting
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the Python version requirement
)


