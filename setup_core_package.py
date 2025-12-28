"""
Setup script for carousel-post-generator-core package

This is a template for creating the public OSS core package.
Copy this to your new public repository and customize as needed.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (
    (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""
)

setup(
    name="social-visual-generator-core",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Core library for generating social media infographics and carousel posts with captions using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robort-gabriel/social-visual-generator-core",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.11",
    install_requires=[
        "langgraph>=0.0.20",
        "langchain>=1.2.0",
        "langchain-openai>=0.0.5",
        "langchain-core>=1.2.5",
        "playwright>=1.40.0",
        "aiohttp>=3.9.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "pydantic>=2.10.4",
        "pydantic-settings>=2.7.0",
        "python-dotenv>=1.0.0",
        "cairosvg==2.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add any CLI commands here if needed
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
