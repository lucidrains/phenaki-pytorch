from setuptools import setup, find_packages

def parse_requirements(filename):
    lines = (line.strip() for line in open(filename))
    return [line for line in lines if line and not line.startswith('#')]

setup(
  name = 'phenaki-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.4.2',
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
  install_requires=parse_requirements('requirements.txt'),
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
