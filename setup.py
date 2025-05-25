from setuptools import setup, find_packages

setup(
    name="haven-vlm-engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "Pillow",
        "numpy",
        "torch",
        "pydantic"
    ],
    author="Haven VLM Engine Team",
    author_email="contact@example.com",
    description="A reusable package for VLM data pipelines and complex processing.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/example/haven-vlm-engine-package", # Replace with actual URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose appropriate license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
