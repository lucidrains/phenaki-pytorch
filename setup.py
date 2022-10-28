from setuptools import setup, find_packages

setup(
  name = 'phenaki-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.7',
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
    'einops>=0.4',
    'ema-pytorch',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'vector-quantize-pytorch'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
