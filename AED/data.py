import pandas as pd


class Data:

	def __init__(self, filename):
        self.filename = filename
        
    def getData(colname):
        dfsample = pd.read_csv("AEDsample.csv")
        return dfsample[colname]

