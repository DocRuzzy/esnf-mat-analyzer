from setuptools import setup, find_packages

setup(
    name="esnf_mat_analyzer",
    version="1.0.0",
    author="DocRuzzy and Contributors",
    author_email="your-email@example.com",
    description="Electrospun Nanofiber Mat Analyzer: Background correction, uniformity, and visualization tools for ESNF mat images.",
    # Robustly read README.md for PyPI/JOSS, fallback if missing
    long_description=(open("README.md", encoding="utf-8").read() if __import__('os').path.exists("README.md") else "See https://github.com/DocRuzzy/esnf-mat-analyzer for documentation."),
    long_description_content_type="text/markdown",
    url="https://github.com/DocRuzzy/esnf-mat-analyzer",
    license="GPLv3",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "Pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "piexif>=1.1.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "esnf-analyzer=esnf_mat_analyzer.main:main",
        ],
    },
    project_urls={
        "Documentation": "https://github.com/DocRuzzy/esnf-mat-analyzer",
        "Source": "https://github.com/DocRuzzy/esnf-mat-analyzer",
        "Tracker": "https://github.com/DocRuzzy/esnf-mat-analyzer/issues",
    },
)
