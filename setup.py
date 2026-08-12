#!/usr/bin/env python
"""stereocell-pipeline: StereoCell 细胞分割流程。

双模块架构:
  模块一 ssDNA 细胞核识别 (训练/推理/无督导)
  模块二 RNA 扩散模型细胞分割 (训练/推理/QC/合胞体评估)
"""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="stereocell-pipeline",
    version="1.1.0",
    description="StereoCell cell segmentation: ssDNA nucleus detection + "
                "RNA-diffusion GMM-EM cell assignment with QC and syncytium scoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="zhangpei",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(include=["scell", "scell.*"]),
    py_modules=["nuclei_train", "nuclei_segment", "cell_train", "cell_segment",
                "run_pipeline", "crop_region"],
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-image>=0.21",
        "tifffile>=2023.7",
        "pandas>=2.0",
        "matplotlib>=3.7",
    ],
    extras_require={
        "h5ad": ["anndata>=0.9"],
        "gpu": ["torch>=2.0"],
        "cellpose": ["cellpose>=3.0"],
        "all": ["anndata>=0.9", "torch>=2.0", "cellpose>=3.0"],
    },
    entry_points={
        "console_scripts": [
            "stereocell-nuclei-train=nuclei_train:main",
            "stereocell-nuclei-segment=nuclei_segment:main",
            "stereocell-cell-train=cell_train:main",
            "stereocell-cell-segment=cell_segment:main",
            "stereocell-run=run_pipeline:main",
            "stereocell-crop=crop_region:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
