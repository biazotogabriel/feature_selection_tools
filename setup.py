from setuptools import setup

setup(
    name='feature_selection_tools',
    version='1.0.0',
    description='Library with func-tools to help when selectiong features',
    long_description = 'Library with func-tools to help when selectiong features',
    author='Gabriel Nuernberg Biazoto',
    author_email='biazotogabriel@gmail.com',
    url='https://github.com/biazotogabriel/feature_selection_tools',
    packages=['feature_selection_tools'],
    license = 'MIT',
    keywords = 'features selection tools',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas==1.4.4',
        'matplotlib==3.5.2'
    ],
)