#!/usr/bin/env python
# coding: utf-8

# # Group J
# 
# ### Group Assignment
# 
# ### Amy Hammond, Ben Rose, Joe Pimblett, Pushkar Waghchoure & Saurabh Giram

# #### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sys


# #### Reading CSV data file, saved as callable dataframe

# In[2]:


dfsample = pd.read_csv("AEDsample.csv")
dfsample # also print type of data


# In[3]:


#Random but Unique sample (seed = 530) 
sample = dfsample.sample(n = 400, random_state = 530)  # n = sample size , random_state sets the seed


# In[4]:


# columns within the sample dataframe
sample.columns 


# #### Generate descriptive statistics that summarise the central tendency, dispersion and shape of a dataset's distribution

# In[5]:


sample.describe()


# #### Dropping/Removing certain rows from the table generated using the describe() function
# 

# In[6]:


A = pd.DataFrame(sample.describe().drop(['count','25%','50%','75%']))
A


# #### Creating dictionary for our relationship key(int) to string of the variables.
# #### Used to convert Numeric User Input to Respective String Values.

# In[7]:


keys = {1 : 'Age', 2: 'LoS', 3: 'noofinvestigation', 4: 'nooftreatment',5: 'noofpatients', 6: 'HRG', 7 : 'DayofWeek'}
hrg_keys = {1:'VB02Z', 2: 'VB03Z', 3: 'VB04Z', 4: 'VB05Z', 5: 'VB06Z', 6: 'VB07Z', 7: 'VB08Z', 8: 'VB09Z', 9: 'VB11Z'}
dow_keys = {1: 'Friday', 2: 'Monday', 3: 'Saturday', 4: 'Sunday', 5: 'Thursday', 6: 'Tuesday', 7:'Wednesday' }


# #### Defining Data Class for data related operations

# In[8]:


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
    


# #### Graph class for plotting graphs and charts

# In[9]:


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
    

    def PlotHistogram(self, xaxis, yaxis, type, inputparam, lowerLimit, upperLimit):
                n, bins, patches = plt.hist(x=self.getColumnData(xaxis), bins=self.getBinsList(lowerLimit,upperLimit,
                                   inputparam), color=self.color, alpha=self.alpha, rwidth=self.rwidth)
                plt.grid(axis='y', alpha=0.75)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                plt.title(type+' '+xaxis)
                maxfreq = n.max()
                # Set a clean upper y-axis limit.
                plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
                
    def PlotHistogramForColumn(self, colName):
                self.getColumnData(colName).hist()
     
    def PlotHistogramForBinSize (self, colName, sizeofbin, xlable, ylable, type):
            n, bins, patches = plt.hist(x=self.getColumnData(colName), bins=sizeofbin, color=self.color, alpha=self.alpha, rwidth=self.rwidth)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel(xlable)
            plt.ylabel(ylable)
            plt.title(type+' '+xlable)

                
    def PlotScatter (self, xaxis, yaxis, type):
                plt.scatter(self.getColumnData(xaxis),self.getColumnData(yaxis), color = self.color)
                plt.title(type+' '+xaxis)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                
    def PlotBar (self, xaxis, yaxis, type):
                plt.bar(self.getColumnData(xaxis),self.getColumnData(yaxis), color = self.color, edgecolor = self.edgecolor)
                plt.title(type+' '+xaxis)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                
    def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                plt.bar(xaxislist, yaxisdata)
                plt.title(xaxis+' vs '+yaxis)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                
    def PlotBoxPlot (self, colname):
                self.getColumnData(colname).plot.box()    
            
    def PlotDisPlot (self, colname):   
            sns.displot(self.getColumnData(colname), color=self.color, bins=100)
                  
    def PlotPieChart (self, colname):
            self.getColumnData(colname).value_counts().plot.pie()
                


# #### Function to convert numeric user Input to respective string values (one input)

# In[10]:


def int_to_str(x):
   # z = 0
    for i in range(1,len(keys)+1):
        if(x == i):
            z = keys[i]   
    return z


# #### Function to convert numeric user Input to respective string values (two inputs)

# In[11]:


def int_to_str2(x,y):
    z = 0
    p = 0
    for i in range(1,len(keys)+1):
        if(x == i):
            z = keys[i]
            for j in range(1,len(keys)+1):
                if( y == j):
                    p = keys[j]
    return z,p


# #### Function to link numeric user input to respective string values (HRG)

# In[12]:


def hrg_values(v):
    for i in range(1,len(hrg_keys)+1):
        if(v == i):
            z = hrg_keys[i]

    return z


# #### Function to link numeric user input to respective string values (HRG) 

# In[13]:


def dow_values(x):
    for i in range(1,len(dow_keys)+1):
        if(x == i):
            z = dow_keys[i]
    return z  


# ### Function to  allow the user to input the patient ID, and return information about this patient from the sample.

# In[14]:


def patient_info():
    
    Patient_Id = input(" Please provide the id of patient ? ")
    Patient_info = sample.loc[sample['ID'] == Patient_Id]
    if Patient_info.empty == True:
        print("Oops! That was no valid number. Try again...")
    else:
        print(Patient_info)


# ### Correlation function for two submitted variables

# In[15]:


def correlation(a,b,data = sample):
    corr =  round(data[a].corr(data[b]) , 3)
    return corr


# #### Function to return a correlation heatmap between all variables 

# In[16]:


def full_correlation(data = sample):
    corr = data.corr()
    ax = sns.heatmap(corr,  vmin=-1 , vmax=1, center=0, cmap="afmhot_r", square=False)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');


# ### Function - offers a variable range (Min, Max) for the previously selected variable. Continuous variables only

# In[17]:


def variable_range(c):

    if(c == "Age"):
         inp4, inp5 = input("Choose 2 numbers between "+A.iloc[2,0].astype(str)+" and "+A.iloc[3,0].astype(str)+", eg. lower bound SPACE upper bound\n").split()
    elif(c == "LoS"):
         inp4, inp5 = input("Choose 2 numbers between "+A.iloc[2,3].astype(str)+" and "+A.iloc[3,3].astype(str)+", eg. lower bound SPACE upper bound\n").split()
    elif(c == "noofinvestigation"):
         inp4, inp5 = input("Choose 2 numbers between "+A.iloc[2,4].astype(str)+" and "+A.iloc[3,4].astype(str)+", eg. lower bound SPACE upper bound\n").split()
    elif(c == "nooftreatment"):
         inp4, inp5 = input("Choose 2 numbers between "+A.iloc[2,5].astype(str)+" and "+A.iloc[3,5].astype(str)+", eg. lower bound SPACE upper bound\n").split()       
    elif(c == "noofpatients"):
         inp4, inp5 = input("Choose 2 numbers between "+A.iloc[2,6].astype(str)+" and "+A.iloc[3,6].astype(str)+", eg. lower bound SPACE upper bound\n").split()

    range_data = sample[(sample[c] >= int(inp4) ) & (sample[c] <= int(inp5))]
    return range_data


# ### Function - offers a variable range (Min, Max) for the previously selected variable for continuous variables
# #### In addition, returning a list of the options for the categorical variables

# In[18]:


def categorical(c,data = sample):
    if(c == "Age"):
        inp1 = int(input("Choose one between "+A.iloc[2,0].astype(str)+" and "+A.iloc[3,0].astype(str)+"\n"))
    
    elif(c == "LoS"):
        inp1 = int(input("Choose one between "+A.iloc[2,3].astype(str)+" and "+A.iloc[3,3].astype(str)+"\n"))
    
    elif(c == "noofinvestigation"):
        inp1 = int(input("Choose one between "+A.iloc[2,4].astype(str)+" and "+A.iloc[3,4].astype(str)+"\n"))
    
    elif(c == "nooftreatment"):
        inp1 = int(input("Choose one between "+A.iloc[2,5].astype(str)+" and "+A.iloc[3,5].astype(str)+"\n"))
    
    elif(c == "noofpatients"):
        inp1 = int(input("Choose one between "+A.iloc[2,6].astype(str)+" and "+A.iloc[3,6].astype(str)+"\n"))
    
    elif(c == "HRG"):
        column_values = data[["HRG"]].values
        unique_values = np.unique(column_values)
        print("Choose one from: ")
        for i in range(len(unique_values)):
            print(str(i+1)+" : "+ unique_values[i])
        inp = int(input())
        inp1 = hrg_values(inp)
   
    elif(c == "DayofWeek"):
        column_values = data[["DayofWeek"]].values
        unique_values = np.unique(column_values)
        for i in range(len(unique_values)):
            print(str(i+1)+" : "+ unique_values[i])
        inp = int(input())
        inp1 = dow_values(inp) 
        
    unique_data = data.loc[data[c] == inp1]  #Retriveing Data according to the instructions.
    return unique_data 


# #### Function to return Histogram, Box, Pie or Dist plots for the chosen variable. 

# In[19]:


def plots(f,j,data = sample):
     
    #object of class Graph
    graphChild = Graph()

    if(j==1):  #HISTOGRAM
        if(f=='Age'):
            inputAge = int(input("Please input size of Age ranges \n {eg. 10 = 10 Years}:"))
            #PlotHistogram(xaxis, yaxis, type, inputparam)
            graphChild.PlotHistogram(f,"Frequency", "Histogram",inputAge, 0, 102)
            
        elif(f == 'LoS'):
            lengthOfstayBuckets = int(input("Please input size of Length of Stay ranges \n {eg . 10 = 10 minutes}:"))
            graphChild.PlotHistogram(f,"Frequency", "Histogram",lengthOfstayBuckets, 0, 500+lengthOfstayBuckets+1)
       
        elif(f == 'nooftreatment'):
            graphChild.PlotHistogramForColumn(f)  
             
        elif(f == 'noofpatients'):        
            graphChild.PlotHistogramForBinSize(f, 20,'No. of Patients', 'Frequency', 'Histogram' )
                 
        elif(f == 'noofinvestigation'):
            #def PlotHistogram(self, colName, sizeofbin, xlable, ylable, type):
            graphChild.PlotHistogramForBinSize(f, 20,'No. of Investigations', 'Frequency', 'Histogram' )
    
        elif(f == 'HRG'):
              graphChild.PlotHistogramForBinSize(f, 9,'HRG', 'Frequency', 'Histogram' )
                
        elif(f == 'DayofWeek'):
            graphChild.PlotHistogramForBinSize(f, 7,'HRG', 'Frequency', 'Histogram' )
            
            
            
    elif(j==2): # Box Plot
            if (f== "Age" or f =="LoS" or f=="nooftreatment" or f=="noofinvestigation" or f=="noofpatients" ):
                graphChild.PlotBoxPlot(f)
            else:
                print("Error: Unable to produce chart for the given parameter")
        
    elif(j==3): # PIE CHART
            if (f == 'noofinvestigation' or f == 'DayofWeek' or f == 'HRG' or f == 'nooftreatment'):
                graphChild.PlotPieChart(f)
            else:
                print("Error: Unable to produce pie chart for continuous data")
        
        
    elif(j==4): # Dist Plot - A Distplot or distribution plot, depicts the variation in the data distribution
          graphChild.PlotDisPlot(f) 


# ### Function - returns Scatter or Box plots for the two chosen variables. 

# In[20]:




def plots_two(f,g,h,data= sample):
    
    graphChild = Graph()
    if(h==1): # SCATTER PLOTS
        
        if(f == 'Age' or g =='Age'):
            if(f == 'LoS' or g == 'LoS'):
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
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
                graphChild.PlotScatter(f or g,g or f, "Scatter Plot")
            
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
                b = (data.groupby('noofinvestigation')['Age'].mean())
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
                a = ("Monday", "tuesdsay",'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
                b = (data.groupby('DayofWeek')['Age'].mean())
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
                a = ("Monday", "tuesdsay",'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
                b = (data.groupby('DayofWeek')['nooftreatment'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
        
        elif(f == 'noofpatients' or g =='noofpatients'):
            
            if(f == 'HRG' or g == 'HRG'):
                a = ("VB02Z", "VB03Z",'VB04Z', 'VB05Z', 'VB06Z', 'VB07Z', 'VB08Z', "VB09Z", "VB11Z")
                b = (data.groupby('HRG')['noofpatients'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
                
            elif(f == 'DayofWeek' or g == 'DayofWeek'):
                a = ("Monday", "tuesdsay",'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
                b = (data.groupby('DayofWeek')['noofpatients'].mean())
                #def PlotBarWithParameters (self,xaxis,yaxis, xaxislist, yaxisdata):
                graphChild.PlotBarWithParameters(f or g,g or f, a, b)
                
        elif(f == 'HRG' or g == 'HRG'): 
            if(f == 'DayofWeek' or g == 'DayofWeek'):
                print("Error: Unable to produce chart for categorical data")


# #### Function to show variables affecting breaches

# In[21]:


def more():
    df = sample.loc[sample['Breachornot']=='breach']
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))
    
    df["HRG"].value_counts().plot.pie(ax = ax1)
    df["DayofWeek"].value_counts().plot.bar(color=["r",'g','b'], ax = ax2)


# ### *Function - asks the user for inputs which call on predefined functions based off responses
# 
# ##### Correlate - correlations between two specific variables or all correlations pairs from the variables
# ##### Plot - return a one or two variable plot, based off inputs
# ##### Range - Ability to subset the sample data set by a speicifc contraint and then use this new sample for analysis 
# ##### Patient Info - Allow the user to input the patient ID, and return information about this patient
# ##### Categorical - Offers a variable range (Min, Max) for the previously selected variable for continuous variables. 
# ####           In addition, returning a list of the options for the categorical variables
# ##### Exit - Return statement as an output for the function, exiting the function. 

# In[23]:


def main(data = sample):
    while(True):
            try: 
                m = input("Select your desired operation \n 1. Correlate 2. Plot 3. Range 4. Patient Info  5. Categorical  6. Breach Info 7. Exit \n Choose between 1-6 \n")  
                if(int(m) == 1): #1. Correlate
                    while(True):
                        try:
                            inp = input("Do you want to Correlate 1. Full Data or  2.Two Variables \n Choose 1 or 2 \n")
                            if(int(inp) == 1):
                                full_correlation()
                            elif(int(inp) == 2):
                                inp1 , inp2 = input("Which TWO variables do you want to compare? \n 1: Age   2:LoS   3: noofinvestigation   4: nooftreatments   5:noofpatients  \n Choose one from 1-5 SPACE Choose different one from 1-5 \n").split()
                                a, b = int_to_str2(int(inp1), int(inp2))
                                print(correlation(a,b,data))
                            else:
                                print("Invalid Input")
                            break
                        except ValueError:
                            print("Incorrect input!")
                        except KeyError:
                            print("Invalid Input: Please Choose Numbers between 1-5 with a SPACE between them.")
                        except TypeError:
                            print("Invalid Input: Please Choose Numbers between 1-5 with a SPACE between them.")
   
                elif(int(m) == 2): # 2. Plot
                    while(True):
                        try:
                            inp = input("Do you want plots for 1 or 2 variables ?\n Choose 1 or 2 \n")
                            if(int(inp) == 1):
                                inp9 = input("Which variable do you want to plot? \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatment 5:noofpatients 6: HRG \7: Day of the Week \n  Choose between 1-7 \n")
                                verify9 = [1,2,3,4,5,6,7]
                                if(int(inp9) in verify9):
                                    x = int_to_str(int(inp9))
                                    inp10 = int(input("Choose one of the following plots:\n 1 : Histogram \n 2 : BoxPlot \n 3 : Pie Chart \n 4 : Dist Plot \n Choose between 1-4 \n"))
                                    print("here?")
                                    plots(x,inp10,data)
                                else:
                                    print("here?")
                                    print("Invalid Input")
                                    main() 
            
                            elif(int(inp) == 2): 
                                inp6 = input("Which variable do you want to plot? \n  1: Age 2: LoS   3: noofinvestigation  4: nooftreatment  5:noofpatients  6: HRG 7. Day of the Week\n Choose between 1-7 \n")
                                verify6 = [1,2,3,4,5,6,7]
                                if(int(inp6) in verify6):
                                    a = int_to_str(int(inp6))
                                
                                    if (a == 'Age'):
                                        ch = int(input("Which Variable do you want to plot Age with \n 2: LoS   3: noofinvestigation  4: nooftreatments  5:noofpatients  6: HRG  7: DayofWeek \n"))
                                        verify = [2,3,4,5,6,7]
                                        if ch not in verify:
                                            print("Invalid Input")
                                            continue
                    
                                    elif (a == 'LoS'):
                                        ch = int(input("Which Variable do you want to plot LoS with \n 1: Age 3: noofinvestigation 4: nooftreatments 5: noofpatients 6: HRG  7: DayofWeek \n"))
                                        verify = [1,3,4,5,6,7]
                                        if ch not in verify:
                                            print("Invalid Input")
                                            continue
                    
                                    elif (a == 'noofinvestigation'):
                                        ch = int(input("Which Variable do you want to plot noofinvestigation with \n 1: Age 2: LoS 4: nooftreatments 5: noofpatients 6: HRG  7: DayofWeek \n"))
                                        verify = [1,2,4,5,6,7]
                                        if ch not in verify:
                                            print("Invalid Input")
                                            continue
                    
                                    elif (a == 'noofpatients'):
                                        ch = int(input("Which Variable do you want to plot noofpatients with \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatments 6: HRG  7: DayofWeek \n"))
                                        verify = [1,2,3,4,6,7]
                                        if ch not in verify:
                                            print("Invalid Input")
                                            continue
                    
                                    elif (a == 'nooftreatment'):
                                        ch = int(input("Which Variable do you want to plot nooftreatment with \n 1: Age 2: LoS 3: noofinvestigation 5: noofpatients 6: HRG  7: DayofWeek \n"))
                                        verify = [1,2,3,5,6,7]
                                        if ch not in verify:
                                            print("Invalid Input")
                                            continue
                                    elif (a == 'HRG'):
                                        ch = int(input("Which Variable do you want to plot HRG with \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatments 5: noofpatients 7: DayofWeek \n"))
                                        verify = [1,2,3,4,5,7]
                                        if ch not in verify:
                                            print("Invalid Input")
                                            continue
                    
                                    elif (a == 'DayofWeek'):
                                        ch = int(input("Which Variable do you want to plot HRG with \n 1: Age 2: LoS 3: noofinvestigation 4: nooftreatments 5: noofpatients 6: HRG \n"))
                                        verify = [1,2,3,4,5,6]
                                        if ch not in verify:
                                            print("Invalid Input")
                                            continue
                                    else:
                                        print("Invalid Input")
                                        continue
                                
                                else:
                                    continue

                                
                                inp8 = int(input("Choose one of the following plots:\n 1 : Scatter Plot \n 2 : Bar Plot\n Choose 1 or 2 \n"))
                                br = int_to_str(ch)
                                plots_two(a,br,inp8,data)
                    
                            break
                        except ValueError:
                            print("Incorrect input!")
    
                elif(int(m) == 3): # 3. Range
                    while(True):
                        try:
                            inp3 = int(input("Which variable do you want to analyse? \n 1: Age    2: LoS   3: noofinvestigation  4: nooftreatments  6: HRG   5: noofpatients 7. Day of the Week \n Choose between 1-7 \n"))
                            verify3 = [1,2,3,4,5,6,7]
                            if(int(inp3) in verify3):
                                v2 = int_to_str(inp3)
                                data_range = variable_range(v2)
                                print(data_range)
            
                                yn = input("Do you want to analyse data for the range selected?   Choose (Y or N) ===> ")
                                if(yn == 'Y' or yn == 'y'):
                                    main(data_range)
                            
                                else:
                                    main()
                                
                        except ValueError:
                            print("Incorrect input!")
        
                elif(int(m) == 4): #4. Patient Info
                    flag = 0
                    while flag == 0:
                        Patient_Id = input(" Please provide the id of patient ? ")
                        Patient_info = data.loc[sample['ID'] == Patient_Id]
                        if Patient_info.empty == True:
                            print("Oops! That was not a valid number. Try again...")
                            continue
                        else:
                            print(Patient_info)
                            flag = 1
        
                elif(int(m) == 5): #5. Categorical
                    while True:
                        try:
                            inp10 = int(input("Which variable do you want to analyse? \n 1: Age   2:LoS   3: noofinvestigation   4: nooftreatments   5:noofpatients  6. HRG   7. Day of the Week \nChoose between 1-7 \n"))
                            v3 = int_to_str(inp10)
                            data_range = categorical(v3)
                            print(data_range)
            
                            yn = input("Do you want to analyse data for the range selected?   Choose(Y or N) ===> ")
                            if(yn == 'Y' or yn == 'y'):
                                main(data_range)
                            break
                        except ValueError:
                            print("Please Input a Number")
                        except UnboundLocalError:
                            print("Invalid Input : Please choose a number between 1-7")
            
                elif(int(m) == 6):
                    more()
            
                elif( int(m) == 7): # 7 Exit
                    print("Thank You!")
                    sys.exit()
        
                else:
                    print("Error: Invalid Input. \n Please Choose between 1-6")
                    main()
                break
            except ValueError:
                print("That is not a valid input. Please try again with a specified option...")

main()


# In[ ]:





# In[ ]:




