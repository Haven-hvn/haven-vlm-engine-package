from setuptools import setup, find_packages

setup(
    name="vlm_engine",
    version="0.3.1",
    description="Advanced Vision-Language Model Engine for content tagging",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="HAVEN Network",
    author_email="officialhavennetwork@gmail.com",
    url="https://github.com/Haven-hvn/haven-vlm-engine-package",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "numpy",
        "torch",
        "torchvision",
        "aiohttp",
        "pyyaml",
        "opencv-python",
        "decord",
        "requests",
        "multiplexer-llm==0.1.4"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
