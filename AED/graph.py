class Graph:
    
    def __init__(self, xaxis = "Not available", yaxis = "Not available", type="Not available", inputparam= "Not available"):
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.inputparam = inputparam
        self.type = type
        self.color ='OliveDrab'
        self.alpha = 0.65
        self.rwidth = 0.85
    
    def getSampleData(self, colName):
        sampleD = Data()
        return sampleD.getData(colName)
        
    def getBinsList(self,lowerLimit, upperLimit, inputparam):
        return np.arange(lowerLimit, upperLimit, inputparam).tolist()     


    def PlotHistogram(self,  xaxis, yaxis, type, inputparam, lowerLimit, upperLimit):
                n, bins, patches = plt.hist(x=self.getSampleData(self.xaxis), bins=self.getBinsList(lowerLimit,upperLimit
                                                                                                    ,self.inputparam),
                                        color=self.color,
                            alpha=self.alpha, rwidth=self.rwidth)

                plt.grid(axis='y', alpha=0.75)
                plt.xlabel(self.xaxis)
                plt.ylabel(self.yaxis)
                plt.title(self.type+' '+self.xaxis)
                maxfreq = n.max()
                # Set a clean upper y-axis limit.
                plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    def PlotScatter (self):
                plt.scatter(self.getSampleData(self.xaxis),self.getSampleData(self.yaxis), color = "#FF51FF")
                plt.title(self.type+' '+self.xaxis)
                plt.xlabel(self.xaxis)
                plt.ylabel(self.yaxis)
                
    def PlotBar (self):
                plt.bar(self.getSampleData(self.xaxis),self.getSampleData(self.yaxis), color = "#FF51FF", edgecolor = "#FFEFDB")
                plt.title(self.type+' '+self.xaxis)
                plt.xlabel(self.xaxis)
                plt.ylabel(self.xaxis)