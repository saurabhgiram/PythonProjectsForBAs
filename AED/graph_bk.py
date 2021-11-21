class Graph:
    
    sampleD = Data("AEDsample.csv")
    
    def __init__(self, xaxis = "Not available", yaxis = "Not available", type="Not available", inputparam= "Not available"):
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.inputparam = inputparam
        self.type = type
        self.color ='OliveDrab'
        self.alpha = 0.65
        self.rwidth = 0.85
    
    def getSampleData(self, colName):
        return sampleD.getColumnData(colName)
        
    def getBinsList(self,lowerLimit, upperLimit, inputparam):
        return np.arange(lowerLimit, upperLimit, inputparam).tolist()  
    

    def PlotHistogram(self,  xaxis, yaxis, type, inputparam, lowerLimit, upperLimit):
                n, bins, patches = plt.hist(x=self.getSampleData(xaxis), bins=self.getBinsList(lowerLimit,upperLimit
                                                                                                    ,inputparam),
                                        color=self.color,
                            alpha=self.alpha, rwidth=self.rwidth)

                plt.grid(axis='y', alpha=0.75)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                plt.title(type+' '+xaxis)
                maxfreq = n.max()
                # Set a clean upper y-axis limit.
                plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
                
    def PlotScatter (self):
                plt.scatter(self.getSampleData(self.xaxis),self.getSampleData(self.yaxis), color = self.color)
                plt.title(self.type+' '+self.xaxis)
                plt.xlabel(self.xaxis)
                plt.ylabel(self.yaxis)
                
    def PlotBar (self):
                plt.bar(self.getSampleData(self.xaxis),self.getSampleData(self.yaxis), color = self.color, edgecolor = "#FFEFDB")
                plt.title(self.type+' '+self.xaxis)
                plt.xlabel(self.xaxis)
                plt.ylabel(self.xaxis)
                
   # def PlotBarWithParameters (self):
              #  a = list(range(0,(A["nooftreatment"]["max"]))+1))
              #  b = (df.groupby[self.xaxis or self.yaxis][self.xaxis or self.yaxis].mean())
               # plt.bar(a,b, color = "#FF51FF", edgecolor= "#7FFF00")
              #  plt.title("Age VS No. of Investigations")
              #  plt.ylabel('Mean Age')
              #  plt.xlabel('No. of Investigations')
                
                
                
                
