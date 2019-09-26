from setuptools import setup,find_packages

setup(name='PyAnalysisTools',
      version='0.1',
      description='PyAnalysisTools',
      url='http://github.com/storborg/funniest',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=find_packages(),  #packages=['PyAnalysisTools'],
      zip_safe=False, install_requires=['scikit-learn', 'pandas'])