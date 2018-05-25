from setuptools import setup, find_packages

# strong_requirements = [
#     "Sphinx==1.6.3",
#     "sphinx-rtd-theme==0.2.4",
#     "nose==1.3.7",
#     "nose-timer==0.7.0",
#     "argparse",
#     "google-cloud==0.27.0",
#     "pandas==0.20.3",
#     "pandas-gbq==0.2.0",
#     "retrying==1.3.3",
#     "mock==2.0.0",
#     "pylint==1.7.2",
#     "Jinja2==2.9.6",
#     "coverage==4.4.1",
#     "lxml==3.8.0",
#     "sqlalchemy==1.1.13",
#     "numpy==1.13.1", # 1.11.1 and 1.12.1 do not work : AttributeError: module 'pandas' has no attribute 'compat'
#     "mysqlclient==1.3.10",
#     "oss2==2.3.4"
# ]

# dependencies_requirements = [
#         "setuptools==34.0.0",
#         "oauth2client==2.2.0",
#         "pyasn1==0.4.2",
#         "requests>=2.18.0",
# ]

all_requirements = [] #strong_requirements + dependencies_requirements

setup(name='RecommenderSystem',
      version='0.0.1',
      description='Recommender System algorithms',
      url='https://github.com/RomainWarlop/RecommenderSystem',
      author='Romain WARLOP',
      author_email='roaminwarlop@gmail.com',
      license='',
      packages=['RecommenderSystem'],
      install_requires=all_requirements,
      #test_suite='nose.collector',
      #tests_require=['nose'],
      zip_safe=False,
      include_package_data=True)
