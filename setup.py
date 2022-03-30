from setuptools import setup

setup(name='src',
      version='1.0.0',
      description='Prediction of survival rate of patients suffering from coronary artery diseases',
      author='chowhp',
      packages=['src'],
      install_requires=['ipython==7.31.0',
                        'ipython-genutils==0.2.0',
                        'ipykernel==6.6.1',
                        'jupyter-client==7.1.0',
                        'jupyter-core==4.9.1'
                        'matplotlib==3.1.2',
                        'nbconvert==6.4.0',
                        'nbformat==5.1.3',
                        'numpy==1.17.4',
                        'pandas==1.3.5',
                        'scikit-learn==1.0.2'])
