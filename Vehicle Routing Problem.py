#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 10:00:08 2021

@author: Saurabh Giram (35493311)

"""
# Import all the necessary libraris

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import ttk
from random import randint

#  Read the file which user has asked for
filename = input(" Please give the file name in the format of I1 or I2...?")
inputfile = filename + '.csv'

# Take the number of iterations from the user for which user want to perform iterations.
numIterations = int(input(" For how many iterations you want to run the solution ?"))
# Take the number of iterations from the user for which user want to perform iterations.

# Read the CSV file and convert it into DataFrame
df_vehicle_routing = pd.read_csv(inputfile)
# print the file data
df_vehicle_routing

# Create customer ID variable and Store Customer Id from the file
customerIDs = df_vehicle_routing['ID']
# Print the customer ID
customerIDs
# Create a list of customer IDs
listCustomerIDs = list(customerIDs)

# Find the Number of Customers 
numCustomers = len(df_vehicle_routing)

# Print the number of customers
numCustomers

# Number of available trucks are:
availabletrucks= int(numCustomers**(1/2))
# Print the number of available trucks
availabletrucks

# Create a truck list based on the available trucks
truckList = [f'Truck {i}' for i in range (1, availabletrucks+1)]
# Print the truck list
truckList

# Allocation of deliveries to the trucks
deliveryallocation = int(numCustomers / availabletrucks)
# Print the alloaction of deliveries to the trucks
deliveryallocation

# Add distances of the customer's location from the depot to the dataframe
df_vehicle_routing['Distance from Depot'] = np.sqrt(df_vehicle_routing["x"]**2 + df_vehicle_routing["y"]**2)


#This is creating the pickle file to store the results 
pklfilename=filename+'.pkl'

class dfUtility:

    #  addDepot function add depot co-ordinate to the dataframe
    def addDepot(df_quadrant=df_vehicle_routing):
        df2 = {'ID': 'Depot', 'x': 0, 'y': 0}
        df_quadrant_withdepot=df_quadrant.append(df2, ignore_index = True)
        return df_quadrant_withdepot

    """
    Create_distance_matrix function calculates the distance between various customer's location and store it in the distance matrix variable.
    """       
    
    def create_distance_matrix(data = df_vehicle_routing, number = numCustomers ):
        distance_matrix = [[0 for i in range(number) ]for j in range(number)] # Creating empty distance matrix to fill in next step
        for i in range(number):
                    for j in range(number):    
                            distance_matrix[i][j]= np.sqrt((data.iloc[i,1] - data.iloc[j,1])**2 + (data.iloc[i,2] - data.iloc[j,2])**2) # Calculate the distance matrix
        return distance_matrix

    """
    create_depot_distancematrix function is for creating depot distance matrix
    """
    def create_depot_distancematrix():
        depotlist = [0 for i in range (numCustomers)] # Creating empty distance matrix to fill in next step
        for i in range (numCustomers):
            depotlist[i]= np.sqrt((df_vehicle_routing.iloc[i,1])**2+(df_vehicle_routing.iloc[i,2])**2) # Calculating the Eucleadean distance
        print(depotlist)
        
    """
    routeLength function will calculate the routelength and alos add the distance between the depot to the first customer and last customer
    """
    def getRouteLength(solution , distance_matrix ,data, number = numCustomers):
        distance = 0 
        for i in range (number-1):
            distance += distance_matrix[solution[i]][solution[i+1]] # calculate the route for given solution
        rootdistance=(data.iloc[solution[-1],3])+(data.iloc[solution[0],3]) # Add the distance between the first and last customer to the depot
        distance+=rootdistance
        return distance


class Solution:
        
    """
    initialSolution function initialize the route path.
    """
    def getInitialSolution(number = numCustomers):
        solution = list(range(number)) 
        np.random.shuffle(solution) # Shuffle the random initial solution
        return solution
        print("This is very intial solution created randomly for all the vehicles" ,solution)
    
    """
    # After reevaluating the random solution we created in initial solution, run below function to get the better and intelligent solution of route.
    # Below better route solution we found out based on customer's quadrant allocation and respective distances
    """
    def getImprovedInitialsol(data=df_vehicle_routing):
        sortedDf=pd.DataFrame() # Create the empty dataframe
        x = np.asarray(df_vehicle_routing["x"]) # Create the array for x co-ordinates
        y = np.asarray(df_vehicle_routing["y"]) # Create the array for y co-ordinates
        df_vehicle_routing['Direction'] = direction = np.rad2deg(np.arctan2(y, x))   # Apply the arc tangent function from numpy and calc the angles and store into the dataframe
        sortedDf=df_vehicle_routing.sort_values(["Direction"], ascending = (True)) # Sort the dataframe
        trucwise_list = list(SolutionUtility.split_list(sortedDf,deliveryallocation)) # Create the vehicle wise list
        return trucwise_list
        
    
    
    """
    hillClimbing hillClimbing function call the swap function to swap between the customer ID in the solution path,
    then call the distance calculator and compare the distance between various paths.
    """
    
    def hillClimbing(solution , distance_matrix ,data=df_vehicle_routing, number = numCustomers):
        iterations = []
        costLine = []
        costLineNew = []
        count=0
        solutionCost = dfUtility.getRouteLength(solution, distance_matrix,data, number) # Calculate the solution cost with current solution
        plt.figure()
        for i in range (numIterations):
            x = np.random.randint(0, number)
            y = np.random.randint(0, number)
            SolutionUtility.SwapOperator(solution, x , y) # swap the customer in solution
            count+=1
            solutionCost_new = dfUtility.getRouteLength(solution, distance_matrix,data, number) # Calculate the new cost
                    
            if i % 100 == 0: # This is for plotting the iterations vs cost line
                iterations = iterations + [i]
                costLine = costLine + [solutionCost]
                costLineNew = costLineNew + [solutionCost_new]
                print ('Hill climbing cost vs iteration graph')            
                Graph.createGraph(iterations,costLine, costLineNew )
            if solutionCost_new <= solutionCost :  # Check if new solution is better or not
                solutionCost = solutionCost_new 
            else:
                SolutionUtility.SwapOperator(solution, x, y) # If not then again swap and calculate the new cost
                count+=1
           
        print(f"Solution cost after {count} swipes and Hillclimbing is {solutionCost}")
        return solution
    
    
    """
    This simulated_annealing function is for calculating and evavulating the better route for vehicles based on Simulated Annealing algorithm
    """
    def simulated_annealing (solution , distance_matrix , data=df_vehicle_routing,number = numCustomers): 
        temp = 100 # Initialaze the temp
        factor = 0.1
        temp_init = temp
        count = 0
        solutionCost = dfUtility.getRouteLength(solution, distance_matrix ,data, number) # Calculate the current cost
        print(solutionCost)
        for i in range(numIterations):
            temp = temp*factor # Reduce the temp
            for j in range (numIterations):
                x = np.random.randint(0, number)
                y = np.random.randint(0, number)
                SolutionUtility.SwapOperator(solution, x , y) # swap the customer id
                count+=1
                solutionCost_new = dfUtility.getRouteLength(solution, distance_matrix, data, number) # Calculate the new cost
                if solutionCost_new <= solutionCost:
                    solutionCost = solutionCost_new            
                else:
                    m = np.random.uniform(0,1)
                    if m < np.exp((solutionCost - solutionCost_new)/temp): # Reduce the probablity
                        solutionCost = solutionCost_new 
                    else:
                        SolutionUtility.SwapOperator(solution, x, y)
                        count+=1
                        
        print(f"Solution cost after {count} swipes and simulated_annealing is {solutionCost}")
        return solution
    
    



    


    

class SolutionUtility:
    """ SwapOperator perform the swap between the customers IDs in the initial solution.
    """        
    def SwapOperator(solution, x , y):
        temp   = solution[x] # Perform the swap
        solution[x] = solution[y]
        solution[y] = temp
    
    """
    The function to the split the route according to the allowable vehicles and their limits
    """
    def split_list(lst, n):  
        for i in range(0, len(lst), n): 
            yield lst[i:i + n]  
            
    """
    The function to get the total distance in order to print into the solution file
    """       
    def getSum(totalDistance):
        sum=0
        for i in range (len(totalDistance)): 
            sum+=totalDistance[i]
        return sum
    
    """ getCustId function returns the customer ID associated with the final solution optimised route
    """

    def getCustId(solution, data=df_vehicle_routing):
        custList= []
        for i in range(len(solution)):
            custList.append(int(data.iloc[solution[i]]['ID']))        # Get the customer id 
        return custList
    """
    This function is checking the feasibility of the route based on the given condition
    """
    def getFeasibility(solution, number= numCustomers):
        feasFirst  = 0
        feasSecond = 0
        if (solution[0] % 2)> 0:
                feasFirst = 1
        elif (solution[availabletrucks -1] % 2) == 0:
                feasSecond = 1
        feas = feasFirst + feasSecond
        return feas

    
class FileBuilder:    
    '''
    Function for creating required formated solution.csv file
    '''
    def createSolutionfile(solutionlist, totaldistancewithdelta, feas ):
        solution =[]
        with open('solution.csv', 'w') as writeFile: # Create a new file 
            print("Final solution is written at :", writeFile.name)
            
        for i in range(len(solutionlist)): 
            with open('solution.csv', 'a') as writeFile:
                solution=solutionlist[i]
                writeFile.write(','.join(str(s)for s in solution))   
                writeFile.write('\n')
        
        with open('solution.csv', 'a') as writeFile: # Write the solution file
            writeFile.write(str(totaldistancewithdelta)) # Write the total distance
            writeFile.write('\n')
            writeFile.write(str(feas))
            
    '''
    Function for creating pickle file
    '''
    def createpicklefile(pklfilename, solutionlist, totaldistancewithdelta, feas):
        try:
            pickle_name = open(pklfilename, "rb") # Read the existing file if present
            prev_run = pickle.load(pickle_name)
            print("Previous result as per pickle file :" ,prev_run)
            pickle_name.close()
        except FileNotFoundError:
            print("File not found!")  
            
        Result = [filename, solutionlist, totaldistancewithdelta, feas]    # if not present then create new one   
        pickle_name = open(pklfilename, "wb")
        pickle.dump(Result,pickle_name) # Store the result in the pickle file
        pickle_name.close()
            
    
    

class Graph:
    
    """
    This function will plot the graph
    """
    def createfourQuadGraph(dataframe_collection, data=df_vehicle_routing):   
        plt.figure(figsize=(10,10)) #Size of Graph
        colorList=[]
        trucknumber=len(dataframe_collection)
        for i in range(trucknumber):
                colorList.append('#%06X' % randint(0, 0xFFFFFF))
        
        for key in dataframe_collection.keys():
            data=dataframe_collection[key]
            truckcolor=colorList[int(key)]
            for i in range(len(data)): 
                        #Connect first point with depot
                        if(i==0):
                           x2 = data.loc[i]['x'] 
                           y2 = data.loc[i]['y']
                           plt.plot([0,x2],[0,y2],color=truckcolor)
                           x1 = data.loc[i]['x'] 
                           y1 = data.loc[i]['y'] 
                           x2 = data.loc[i+1]['x'] 
                           y2 = data.loc[i+1]['y']            
                           plt.plot([x1,x2],[y1,y2], color=truckcolor)
                        #Connect last point with depot
                        elif(i==len(data)-1):
                            x1 = data.loc[i]['x'] 
                            y1 = data.loc[i]['y']
                            plt.plot([x1,0],[y1,0], color=truckcolor, label=f'truck{key}'.format(key)) 
                        #connect points on the path
                        else:
                            x1 = data.loc[i]['x'] 
                            y1 = data.loc[i]['y'] 
                            x2 = data.loc[i+1]['x'] 
                            y2 = data.loc[i+1]['y']            
                            plt.plot([x1,x2],[y1,y2], color=truckcolor) 
                    
       
        axis = plt.gca() #3 Get Current Axis
        plt.Button
        plt.plot(axis.get_xlim(),[0,0],'k--') #4 X Axis Plots Line Across
        plt.plot([0,0],axis.get_ylim(),'k--') #5 Y Axis Plots Line Across
        plt.ylabel('y') # Label to the axis
        plt.xlabel('x') # Label to the axis
        plt.title('Vehicle Routing Graph') # Title of the plot
        plt.grid()
        plt.legend()
        plt.show()
        
    def createGraph(iterations,costLine, costLineNew ):
            plt.plot(iterations,costLine)
            plt.plot(iterations,costLineNew)
            plt.title('Hill climbing cost vs iteration graph ')
            plt.ylabel('Cost') # Label to the axis
            plt.xlabel('Iterations')
            plt.grid()
            plt.show()

""" main function is where it will call different functions to find the optimsed route path and return the optimsed distance
"""
def main():     
    totalDistance = [] # Initializing the total distance as 0
    totalFeasibility =0 # Initilailze the feasibility as zero
    totaldistancewithdelta = 0  
    solutionlist=list() # Generate the solution list
    #Get improved initial solution 
    truckCustList = Solution.getImprovedInitialsol()# Calling the Intelligent solution
    dataframe_collection = {} 
    
    #per truck calculations
    for i in range (len(truckCustList)):
        print('Truck', i)
        print(truckCustList[i])
        dfq=pd.DataFrame(truckCustList[i])# Calling the truck list
        number = len(dfq) 
        # Genearte distance matrix 
        distance_matrix=dfUtility.create_distance_matrix(data = dfq, number=number) # Fetching the result from distance matrix
        solution = Solution.getInitialSolution(number)
        
        # Apply hillclimbing
        solution=Solution.hillClimbing(solution , distance_matrix ,data=dfq,number=number) # Calling the hill climbing
        
        #Apply simulated annealing
        solution=Solution.simulated_annealing(solution , distance_matrix ,data=dfq,number=number)
        
        totalDistance.append(dfUtility.getRouteLength(solution, distance_matrix ,dfq, number)) # Append the total distance
        dfq = dfq.reset_index(drop=True)
        solution=SolutionUtility.getCustId(solution, data=dfq) 
        solutionlist.append(solution) 
        
        #Add all dataframes to single list to plot the graph
        dataframe_collection[i] = pd.DataFrame(dfq)
        totalFeasibility += SolutionUtility.getFeasibility(solution, number= numCustomers)
        print('Final customer id path :',solution, totalDistance[i])

    totalDistance.sort()
    print ('***Final Solution ***')
    print ('Total distance without delta : ', SolutionUtility.getSum(totalDistance))    
    totaldistancewithdelta=SolutionUtility.getSum(totalDistance)+abs(totalDistance[0]-totalDistance[-1]) # Add the delta distance to the total distance
    print('Total distance with delta :',totaldistancewithdelta)
    print("Feasibility of this solution is :" , totalFeasibility)
    FileBuilder.createSolutionfile(solutionlist, totaldistancewithdelta, totalFeasibility)
    FileBuilder.createpicklefile(pklfilename, solutionlist, totaldistancewithdelta, totalFeasibility) # Calling the pickle function
    print(f'Graph for file {filename} with {numIterations} iterations is plotted')
    Graph.createfourQuadGraph(dataframe_collection, data=dfq) # Calling the function to plot the graph
    print("Run Completed Successfully")

"""
Main function
"""   
if __name__ == "__main__":
    main()
 
    
