from setuptools import setup, find_packages

# Note: This package uses CPU-only PyTorch by default to minimize installation size.
# For GPU support, install with: pip install "vlm_engine[gpu]"
#
# Or manually install PyTorch with CUDA:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

setup(
    name="vlm_engine",
    version="0.9.2",
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
        "multiplexer-llm==0.2.3"
    ],
    extras_require={
        # GPU support - installs CUDA-enabled PyTorch
        # Note: Users should uninstall CPU torch first: pip uninstall torch torchvision
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
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
