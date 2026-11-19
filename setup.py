"""PerceptGuard-AI: Setup configuration for editable installs."""

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="perceptguard-ai",
        version="1.0.0",
        packages=find_packages(include=["src", "src.*", "config", "config.*"]),
    )
