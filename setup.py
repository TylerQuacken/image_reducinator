from setuptools import setup
setup(
   name='image_reducinator',
   version='0.0.1',
   py_modules=['clusterinator',],
   license='MIT',
   package_dir={'': 'src'},
   description='Reduces the domain of images using k-means clustering',
   url="https://github.com/TylerQuacken/image_reducinator",
   author="Tyler Quackenbush",
   author_email="tylerquacken@gmail.com",
   install_requires = [
       "numpy",
   ],
   extras_require = {
       "dev": [
           "pytest>=3.7",
       ],
   },
)