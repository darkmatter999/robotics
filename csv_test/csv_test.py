import numpy as np
import pandas as pd

#'names' must be explicitly set to true in order to be able to index the column names
#the test .csv file has been created in OpenOffice Calc
csv_np = np.genfromtxt('csv_test.csv', dtype=None, delimiter=',', names=True)

#print (csv_np)

#return only the items in the first column (named 'col1')
#print (csv_np['col1'])


for i in csv_np['col1']:
    #loop through the 'col1' index (field name)
    print (i)
    #integers are read as integers because dtype in above 'genfromtxt' is set to None
    print (type(i))

'''
Numpy arrays must always be of homogeneous datatype (e.g. all int's or all float's, but not float and string together etc.), so 'genfromtxt'
loads the columns as tuple constructs. An ND (1-D, 2-D etc.) array becomes an array of tuples. 
An alternative is reading the .csv file from Pandas in a more concise manner and handle the data in the csv without first converting to a 
Numpy array.
'''

csv_pandas = pd.read_csv('csv_test.csv')
#print (csv_pandas)

#return only the items in the first column (named 'col1')
#print (csv_pandas['col1'])

#print (csv_pandas.head(2))
#print (type(csv_pandas))

for i in csv_pandas['col1']:
    #loop through the 'col1' index (field name)
    print (i)
    #Pandas reads the entries correctly as integers
    print (type(i))

#It is also possible to convert Pandas data frame to Numpy array
#csv_pandas_np = csv_pandas.to_numpy()
#print (csv_pandas_np)
