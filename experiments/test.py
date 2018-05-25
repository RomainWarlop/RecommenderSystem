from pyspark import SparkContext
sc = SparkContext(master, app_name, pyFiles=['/home/romain/Documents/PhD/HOALS/HOALS.py'])
from HOALS import HOALS