from setuptools import setup
from setuptools import find_packages

setup(
    name='farmer',
    version='1.2.6',
    description='Auto Machine Learning for Doctor',
    author='Hiroki Matsuzaki',
    author_email='1234defgsigeru@gmail.com',
    url='https://github.com/NCC-AI/apps',
    download_url='',
    license='Apache 2.0',
    install_requires=[
        'segmentation-models>=1.0.0',
        'albumentations>=0.4.3',
        'optuna>=1.0.0'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow keras machine deep learning',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'Godfarmer=farmer.api.main:fit'
        ],
    },
)
