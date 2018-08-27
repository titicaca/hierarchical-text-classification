from __future__ import print_function
from setuptools import setup, find_packages

setup(
    name="hierarchical-text-classification",
    version="0.1.0",
    author="Yang, Fangzhou",
    author_email="",
    description="Implementation of Hierarchical Text Classification",
    long_description=open("README.md").read(),
    license="MIT",
    url="https://github.com/titicaca/hierarchical-text-classification",
    packages=find_packages(),
    entry_points={
        # 'console_scripts': [
        #     'ternary = ternary.__main__:main'
        # ]
    },
    data_files=[
                ('doc', ['README.md'])
                ],
    include_package_data=True,
    classifiers=[
        "Environment :: Web Environment",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache License',
        'Natural Language :: Chinese',
        'Operating System :: MacOS',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Topic :: NLP',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
            'jieba>=0.39',
            'gensim>=3.4.0',
            'scikit-learn>=0.19.2',
            'pandas>=0.23.3',
            'numpy>=1.14.3'
        ],
    zip_safe=True,
)
