"""
-- Created by: Ashok Kumar Pant
-- Created on: 11/13/20
"""
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []
try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except IOError as e:
    print(e)

test_requirements = []

setup(
    name='gptservice',
    version='1.0',
    description="",
    long_description=readme + '\n',
    author="Ashok Kumar Pant",
    author_email='ashok@treeleaf.ai',
    url='',
    packages=find_packages(),
    namespace_packages=[],
    package_dir={},
    package_data={},
    install_requires=requirements,
    license="",
    zip_safe=False,
    keywords='',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
        'Programming Language :: Python :: 3.8'
    ],
    test_suite='tests',
    tests_require=test_requirements
)
