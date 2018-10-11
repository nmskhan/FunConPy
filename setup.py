from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='funconpy',
      version='0.1',
      description='Resting State fMRI Functional Connectivity Pipeline',
      long_description=readme(),
      classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: OSI Approved :: GNU 3.0',
        'Programming Language :: Python :: 3.6',
        'Topic :: rsfMRI Prerocessing :: Neuroscience',
      ],
      keywords='fMRI preprocessing functional connectivity',
      url='http://github.com/vascosa/FunConPy',
      author='Vasco Sa',
      license='GPL 3.0',
      packages=['funconpy'],
      dependency_links=['https://github.com/ANTsX/ANTsPy']
      install_requires=[
          'numpy',
          'nibabel',
          'scipy',
          'networkx',
          'nipype',
          'nilearn',
          'pandas',
          'numpy',
          'scikit-image',
          'webcolors',
          'matplotlib',
          'plotly',
          'ants'
          
      ],
      include_package_data=True,
      zip_safe=False)
