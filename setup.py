from setuptools import setup, find_packages

setup(
  name = 'phenaki-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.49',
  license='MIT',
  description = 'Phenaki - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/phenaki-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanisms',
    'text-to-video'
  ],
  install_requires=[
    'accelerate',
    'beartype',
    'einops>=0.6',
    'ema-pytorch>=0.1.1',
    'opencv-python',
    'pillow',
    'numpy',
    'torch>=1.6',
    'torchtyping',
    'torchvision',
    'transformers',
    'tqdm',
    'vector-quantize-pytorch>=0.10.14'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
