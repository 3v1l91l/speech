from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='example to run keras on gcloud ml-engine',
      author='Misha Zhukov',
      author_email='misha.zhukov@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py',
          'librosa',
          'tqdm',
          'numpy',
          'pandas'
      ],
      zip_safe=False)
