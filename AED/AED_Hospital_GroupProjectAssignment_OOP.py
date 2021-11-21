#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


keys = {1 : 'Age', 2: 'LoS', 3: 'noofinvestigation', 4: 'nooftreatment',5: 'noofpatients', 6: 'HRG', 7 : 'DayofWeek'}
hrg_keys = {1:'VB02Z', 2: 'VB03Z', 3: 'VB04Z', 4: 'VB05Z', 5: 'VB06Z', 6: 'VB07Z', 7: 'VB08Z', 8: 'VB09Z', 9: 'VB11Z'}
dow_keys = {1: 'Friday', 2: 'Monday', 3: 'Saturday', 4: 'Sunday', 5: 'Thursday', 6: 'Tuesday', 7:'Wednesday' }


# In[3]:


def single_ifs(x):
    for i in range(1,len(keys)+1):
        if(x == i):
            z = keys[i]
    return z


# In[4]:


class Data:

    def getColumnData(self, colname):
        dfsample=pd.read_csv("AEDsample.csv")
        return dfsample[colname]
    
    def getDfsampleData(self):
        dfsample=pd.read_csv("AEDsample.csv")
        return dfsample
    
    def getUpperLimit(self):
        A = pd.DataFrame(self.getSampleData().describe().drop(['count','25%','50%','75%']))
        return  A["Age"]["max"]
    
    


# In[5]:



def ifs(x,y):
    for i in range(1,len(keys)+1):
        if(x == i):
            z = keys[i]
            for j in range(1,len(keys)+1):
                if( y == j):
                    p = keys[j]
    print(z,p)
    return z,p


# In[6]:


def hrg_values(v):
    for i in range(1,len(hrg_keys)+1):
        if(x == i):
            z = keys[i]
    return z


# In[7]:


def dow_values(z):
    
    for i in range(1,len(dow_keys)+1):
        if(x == i):
            z = keys[i]
    return z  


# In[8]:


dfsample = pd.read_csv("AEDsample.csv")
dfsample # also print type of data


# In[9]:


def trying(choice):
    if (choice == 'Age'):
        ch = input("Which Variable do you want to plot Age with \n 2: LoS   3: noofinvestigation  4: nooftreatments  5:noofpatients  6: HRG  7: DayofWeek \n")
    
    elif (choice == 'LoS'):
        ch = input("Which Variable do you want to plot LoS with \n 1: Age 3: noofinvestigation 4: nooftreatments 5: noofpatients 6: HRG  7: DayofWeek \n")
    
    elif (choice == 'noofinvestigation'):
        ch = input("Which Variable do you want to plot noofinvestigation with \n 1: Age 2: LoS 4: nooftreatments 5: noofpatients 6: HRG  7: DayofWeek \n")

    elif (choice == 'noofpatients'):
        ch = input("Which Variable do you want to plot noofpatients with \n 1: Age 2: LoS 3: noofinvestigation 5: noofpatients 6: HRG  7: DayofWeek \n")
    
    elif (choice == 'noofpatients'):
        ch = input("Which Variable do you want to plot noofpatients with \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatments 6: HRG  7: DayofWeek \n")
    
    elif (choice == 'HRG'):
        ch = input("Which Variable do you want to plot HRG with \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatments 5: noofpatients 7: DayofWeek \n")
    
    elif (choice == 'DayofWeek'):
        ch = input("Which Variable do you want to plot HRG with \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatments 5: noofpatients 6: HRG \n")

    return int(ch)


# In[10]:


sample = dfsample.sample(n = 400, random_state = 530)  # n = sample size , random_state sets the seed


# In[11]:


sample.columns # columns within the sample dataframe


# In[12]:


sample.describe()


# In[13]:


A = pd.DataFrame(sample.describe().drop(['count','25%','50%','75%']))
print(A["Age"]["max"])
A


# In[14]:


type(A.iat[2,0])


# In[15]:


def patient_info():
    
    Patient_Id = input(" Please provide the id of patient ? ")
    Patient_info = sample.loc[sample['ID'] == Patient_Id]
    print(Patient_info)


# In[16]:


def correlation(a,b,data = sample):
    corr =  round(data[a].corr(data[b]) , 3)
    return corr


# In[17]:


def full_correlation(data = sample):
    corr = sample.corr()
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');


# In[18]:


class Graph:
    
    def __init__(self, xaxis = "Not available", yaxis = "Not available", type="Not available", inputparam= "Not available"):
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.inputparam = inputparam
        self.type = type
        self.color ='OliveDrab'
        self.alpha = 0.65
        self.rwidth = 0.85
        self.edgecolor="#1E90FF"
    
    def getColumnData(self, colName):
        sampleD = Data()
        return sampleD.getColumnData(colName)
    
    def getSampleData(self):
        sampleD = Data()
        return sampleD.getDfsampleData()
        
    def getBinsList(self,lowerLimit, upperLimit, inputparam):
        return np.arange(lowerLimit, upperLimit, inputparam).tolist()  
    

    def PlotHistogram(self,  xaxis, yaxis, type, inputparam, lowerLimit, upperLimit):
                n, bins, patches = plt.hist(x=self.getColumnData(xaxis), bins=self.getBinsList(lowerLimit,upperLimit,
                                   inputparam), color=self.color, alpha=self.alpha, rwidth=self.rwidth)
                plt.grid(axis='y', alpha=0.75)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                plt.title(type+' '+xaxis)
                maxfreq = n.max()
                # Set a clean upper y-axis limit.
                plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

                
    def PlotScatter (self, xaxis, yaxis, type):
                plt.scatter(self.getColumnData(xaxis),self.getColumnData(yaxis), color = self.color)
                plt.title(type+' '+xaxis)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                
    def PlotBar (self, xaxis, yaxis, type):
                plt.bar(self.getColumnData(xaxis),self.getColumnData(yaxis), color = self.color, edgecolor = self.edgecolor)
                plt.title(type+' '+xaxis)
                plt.xlabel(xaxis)
                plt.ylabel(xaxis)
                
    def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                plt.bar(xaxislist, yaxisdata)
                plt.title(xaxis+' vs '+yaxis)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                
    def PlotBoxPlot (self, colname):
                self.getColumnData(colname).plot.box()    
            
    def PlotDistPlot (self, colname):   
            sns.distplot(self.getColumnData(colname), color=self.color, bins=100)
                


# In[19]:


def variable_range(c):

    if(c == "Age"):
         inp4, inp5 = input("Choose between "+A.iloc[2,0].astype(str)+" and "+A.iloc[3,0].astype(str)+"\n").split()
    elif(c == "LoS"):
         inp4, inp5 = input("Choose between "+A.iloc[2,3].astype(str)+" and "+A.iloc[3,3].astype(str)+"\n").split()
    elif(c == "noofinvestigation"):
         inp4, inp5 = input("Choose between "+A.iloc[2,4].astype(str)+" and "+A.iloc[3,4].astype(str)+"\n").split()
    elif(c == "nooftreatment"):
         inp4, inp5 = input("Choose between "+A.iloc[2,5].astype(str)+" and "+A.iloc[3,5].astype(str)+"\n").split()       
    elif(c == "noofpatients"):
         inp4, inp5 = input("Choose between "+A.iloc[2,6].astype(str)+" and "+A.iloc[3,6].astype(str)+"\n").split()

    range_data = sample[(sample[c] >= int(inp4) ) & (sample[c] <= int(inp5))]
    return range_data


# In[20]:


def categorical(c,data = sample):
    if(c == "Age"):
        sg = int(input("Choose one between "+A.iloc[2,0].astype(str)+" and "+A.iloc[3,0].astype(str)+"\n"))
    
    elif(c == "LoS"):
        sg = int(input("Choose one between "+A.iloc[2,3].astype(str)+" and "+A.iloc[3,3].astype(str)+"\n"))
    
    elif(c == "noofinvestigation"):
        sg = int(input("Choose one between "+A.iloc[2,4].astype(str)+" and "+A.iloc[3,4].astype(str)+"\n"))
    
    elif(c == "nooftreatment"):
        sg = int(input("Choose one between "+A.iloc[2,5].astype(str)+" and "+A.iloc[3,5].astype(str)+"\n"))
    
    elif(c == "noofpatients"):
        sg = int(input("Choose one between "+A.iloc[2,6].astype(str)+" and "+A.iloc[3,6].astype(str)+"\n"))
    
    elif(c == "HRG"):
        column_values = data[["HRG"]].values
        unique_values = np.unique(column_values)
        print("Choose one from: ")
        for i in range(len(unique_values)):
            print(str(i+1)+" : "+ unique_values[i])
        inp = int(input())
        sg = hrg_values(inp)
   
    elif(c == "DayofWeek"):
        column_values = data[["DayofWeek"]].values
        unique_values = np.unique(column_values)
        for i in range(len(unique_values)):
            print(str(i+1)+" : "+ unique_values[i])
        inp = int(input())
        sg = dow_values(inp) 
        
    unique_data = data.loc[data[c] == sg]
    return unique_data


# In[31]:


def plots(f,j,data = sample):

    #object of class Graph - with variables x-axis,y-axis, type of graph and bin input parameter
    # graphChild = Graph(f,"Frequency", "Histogram",inputAge) 
    graphChild = Graph() 
    
    #Methodto plot a histogram with  x-axis,y-axis, type of graph and bin input parameter, lower and upper limit of bin
           
    if(j==1):  #HISTOGRAM
        if(f=='Age'):
            inputAge = int(input("Length of age range of interest:"))
            #PlotHistogram(xaxis, yaxis, type, inputparam)
            graphChild.PlotHistogram(f,"Frequency", "Histogram",inputAge, 0, 102)
            
        elif(f == 'LoS'):
            lengthOfstayBuckets = int(input("Input size of buckets for Length of Stay:):"))
            graphChild.PlotHistogram(f,"Frequency", "Histogram",lengthOfstayBuckets, 0, 500+lengthOfstayBuckets+1)
            
        elif(f == 'noofinvestigation'):            
            RangeOfInterest = int(input("Length of investigation range of interest:"))            
            graphChild.PlotHistogram(f,"Frequency", "Histogram",RangeOfInterest,0,69)
                        
        elif(f == 'nooftreatment'):
            numberOfTreatments = int(input("Length of treatment range of interest:"))            
            graphChild.PlotHistogram(f,"Frequency", "Histogram",numberOfTreatments,0,4)
            
        elif(f == 'noofpatients'):
            numberOfPatients = int(input("Number of patients for which you want to find out frquency:")) 
            graphChild.PlotHistogram(f,"Frequency", "Histogram",numberOfPatients,0,69)

        elif(f == 'HRG'):
            print("Error")
        elif(f == 'DayofWeek'):
            print("Error")
            
    elif(j==2): # Box Plot
        graphChild.PlotBoxPlot(f)      
        
        
    elif(j==3): # PIE CHART
        if(f=='Age'):
            print("Error")
        elif(f == 'LoS'):
            print("Error")
        elif(f == 'noofinvestigation'):
            print("Error")
        elif(f == 'nooftreatment'):
            print("Error")
        elif(f == 'noofpatients'):
            print("Error")
        elif(f == 'HRG'):
            print("Error")
        elif(f == 'DayofWeek'):
            print("Error")
               
        
    elif(j==4): # Dist Plot - A Distplot or distribution plot, depicts the variation in the data distribution
        graphChild.PlotDistPlot(f)            
        
       


# In[22]:


#def plots_recommendation():
    
    


# In[23]:


def plots_simple(f,g,h,data= sample):
    
    graphChild = Graph() 
    if(h==1): # SCATTER PLOTS
        
        if(f == 'Age' or g =='Age'):
            if(f == 'LoS' or g == 'LoS'):
                print("Error")
                  
            elif(f == 'noofinvestigation' or g == 'noofinvestigation'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
                
            elif(f == 'nooftreatment' or g == 'nooftreatment'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
            elif(f == 'HRG' or g == 'HRG'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
                
        elif(f == 'LoS' or g =='LoS'):
            if(f == 'noofinvestigation' or g == 'noofinvestigation'):
                   graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
                
            elif(f == 'nooftreatment' or g == 'nooftreatment'):
                print("Error")
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
            elif(f == 'HRG' or g == 'HRG'):
                   graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
                
        
        elif(f == 'noofinvestigation' or g =='noofinvestigation'):
            if(f == 'nooftreatment' or g == 'nooftreatment'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
                
            elif(f == 'noofpatients' or g == 'noofpatients'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
            elif(f == 'HRG' or g == 'HRG'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
                
        elif(f == 'nooftreatment' or g =='nooftreatment'):
            
            if(f == 'noofpatients' or g == 'noofpatients'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
            elif(f == 'HRG' or g == 'HRG'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                   graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
        
        elif(f == 'noofpatients' or g =='noofpatients'):
            
            if(f == 'HRG' or g == 'HRG'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
                
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
                
        elif(f == 'HRG' or g == 'HRG'): 
            if(f == 'DayofWeek' or g == 'DayofWeek'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
        
    elif(h==2): #BAR PLOTS
        
        if(f == 'Age' or g =='Age'):
            if(f == 'LoS' or g == 'LoS'): 
                graphChild.PlotBar(f or g,g or f, "Bar Plot")
                
            elif(f == 'noofinvestigation' or g == 'noofinvestigation'):                  
                a = list(range(0,7))
                b = (dfsample.groupby('noofinvestigation')['Age'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
            
            elif(f == 'nooftreatment' or g == 'nooftreatment'):
                   graphChild.PlotBar(f or g,g or f, "Bar Plot")
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                    graphChild.PlotBar(f or g,g or f, "Bar Plot")
            
            elif(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['Age'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters('HRG', 'Age', a, b)
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                graphChild.PlotBar(f or g,g or f, "Bar Plot")
                
        
        elif(f == 'LoS' or g =='LoS'):
            if(f == 'noofinvestigation' or g == 'noofinvestigation'):
                a = list(range(0,7))
                b = (data.groupby('noofinvestigation')['LoS'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
                
            elif(f == 'nooftreatment' or g == 'nooftreatment'):
                a = list(range(0,4))
                b = (data.groupby('nooftreatment')['LoS'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                a = list(range(0,69))
                b = (data.groupby('noofpatients')['LoS'].mean())
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
            
            elif(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['LoS'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                a = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
                b = (data.groupby('DayofWeek')['LoS'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
                
        
        elif(f == 'noofinvestigation' or g =='noofinvestigation'):
            if(f == 'nooftreatment' or g == 'nooftreatment'):
                a = list(range(0,7))
                b = (data.groupby('noofinvestigation')['nooftreatment'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
            
            elif(f == 'noofpatients' or g == 'noofpatients'):
                a = list(range(0,7))
                b = (data.groupby('noofinvestigation')['noofpatients'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
            
            elif(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['noofinvestigation'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                a = ("Monday", "tuesdsay",'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
                b = (data.groupby('DayofWeek')['noofinvestigation'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
                
        elif(f == 'nooftreatment' or g =='nooftreatment'):
            
            if(f == 'noofpatients' or g == 'noofpatients'):
                   graphChild.PlotBar(f or g,g or f, "Bar Plot")
                
            elif(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['nooftreatment'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                graphChild.PlotBar(f or g,g or f, "Bar Plot")
        
        elif(f == 'noofpatients' or g =='noofpatients'):
            
            if(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['noofpatients'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
            
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                   graphChild.PlotBar(f or g,g or f, "Bar Plot")
        elif(f == 'HRG' or g == 'HRG'): 
            if(f == 'DayofWeek' or g == 'DayofWeek'):
                graphChild.PlotBar(f or g,g or f, "Bar Plot")
        
        
    elif(h==3):
        return data[f].hist()


# In[40]:


def main(data = sample):

    try:  
        m = int(input("choose \n 1. Correlate 2. Plot 3. Range 4. Patient Info  5. Categorical \n"))

        if(m == 1):
            zzz = int(input("Do you want to correlate 1. Full Data or  2.Two Variables \n"))
            if(zzz == 1):
                full_correlation(data)
                if(zzz == 2):
                    inp1 , inp2 = input("Which TWO varibles do you want to compare? \n 1: Age   2:LoS   3: noofinvestigation   4: nooftreatments   5:noofpatients 7. Day of the Week  \n").split()
                    a, b = ifs(int(inp1), int(inp2))
                    print(correlation(a,b,data))

        elif(m == 2):

                    zzz = int(input("Do you want plots for 1 or 2 variables ?\n"))
                    if(zzz == 1):
                        inp9 = input("Which variable do you want to plot? \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatment 5:noofpatients 6: HRG\n")
                        x = single_ifs(int(inp9))
                        inp10 = int(input("Choose one of the following plots:\n 1 : Histogram \n 2 : BoxPlot \n 3 : Pie Chart \n 4 : Dist Plot \n"))
                        plots(x,inp10,data)

                    elif(zzz == 2): 
                        inp6 = input("Which variable do you want to plot? \n  1: Age 2: LoS   3: noofinvestigation  4: nooftreatment  5:noofpatients  6: HRG 7. Day of the Week\n")
                        a = single_ifs(int(inp6))
                        v1 = trying(a)
                        inp8 = int(input("Choose one of the following plots:\n 1 : Scatter Plot \n 2 : Bar Plot\n"))
                        br = single_ifs(v1)
                        plots_simple(a,br,inp8,data)

        elif(m == 3):
                        inp3 = int(input("Which variable do you want to analyse? \n 1: Age    2: LoS   3: noofinvestigation  4: nooftreatments  6: HRG   5: noofpatients 7. Day of the Week \n"))
                        v2 = single_ifs(inp3)
                        data_range = variable_range(v2)
                        print(data_range)

                        yn = input("Do you want to anylyse data for the range selected?   (Y/N) ===> ")
                        if(yn == 'Y' or yn == 'y'):
                            main(data_range)

        elif(m == 4):
                            Patient_Id = input(" Please provide the id of patient ? ")
                            Patient_info = data.loc[sample['ID'] == Patient_Id]
                            if Patient_info.empty == True:
                                print("Oops! That was no valid id. Try again...")
                                raise exception invalidId 
                            else:
                                print(Patient_info)

        elif(m == 5):
                                inp10 = int(input("Which variable do you want to analyse? \n 1: Age   2:LoS   3: noofinvestigation   4: nooftreatments   5:noofpatients  6. HRG   7. Day of the Week \n"))
                                v3 = single_ifs(inp10)
                                data_range = categorical(v3)
                                print(data_range)

                                yn = input("Do you want to anylyse data for the range selected?   (Y/N) ===> ")
                                if(yn == 'Y' or yn == 'y'):
                                    main(data_range)
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")
        main(data = sample)
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")
        main(data = sample)


main()


# In[ ]:




