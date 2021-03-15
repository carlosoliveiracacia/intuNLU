from setuptools import setup, find_packages

setup(
    name="intunlu",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'datasets==1.4.1',
        'pytorch-lightning==1.2.3',
        'rouge-score==0.0.4',
        "sentencepiece==0.1.95",
        "transformers==4.3.3",
        "torch==1.8.0"
    ],
    python_version='3.6',
)
