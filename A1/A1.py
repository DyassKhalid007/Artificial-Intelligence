"""
   @Author:Dyass Khalid 20-10-0004
   Date: Sunday 24th Feb
"""
"""
   Wall following using genetic algorithm
"""
import random
import math

class GA(object):
    def __init__(self):
        self.pop = 20 #initial population
        self.row = 5
        self.col = 5
        self.population = []#pop move lists for each individual in population to move 
        self.grid = [[]] #the grid which our individuals have to follow
        self.gridInitialization()#function to initialize grid
        self.cycles = 300#number of generations the algorithms has to run
        self.fitness = []#contain results from fitness function
        self.initializePop()
        self.simulation()
        
        #self.selection(self.population[:])
        
    def simulation(self):
        for i in range(self.cycles):
            self.fitnessPopulation(self.population[:])
            self.population = self.selection(self.population[:])
            self.crossover()#plus mutatuin
            self.population = self.selection(self.population[:])
            #print("scores are",self.fitnessPopulation(self.population))
            
        print("Final scores are",self.fitnessPopulation(self.population))
        
       
            
    def gridInitialization(self):
        """function to initialize the grid as asked in the assignment"""
        self.grid = [
        [1,1,1,1,1,1],
        [1,0,0,0,0,1],
        [1,0,0,0,1,2],
        [1,0,0,0,1,2],
        [1,0,0,0,0,1],
        [1,1,1,1,1,1]
        ]
    def initializePop(self):
        """function to initiazlize population with moves 0,1,2,3
           0 means do nothing
           1 means turn right 
           2 means turn left
           3 means move forward
        """
        for i in range(self.pop):
            samples = []
            for j in range(self.cycles):
                samples=samples+[random.randint(0,3)]
            self.population.append(samples[:])
            
    def printGrid(self):
        """function to print the grid"""
        for i in self.grid:
            print(*i)
    def printPop(self):
        for i in self.population:
            print(*i)
    def selection(self,population):
        result = self.fitnessPopulation(population)
        #print("Result:",result)
        total = sum(result)
        newList = [round(i/total*self.pop,2)  for i in result]
        #print(newList)
        newPopulation = []
        length = len(newList)
        for i in range(length):
            if len(newPopulation)<self.pop:
                #print(newList)
                val = max(newList)
                idx = newList.index(val)
                for j in range(math.ceil(val)):
                    newPopulation.append(population[idx][:])
                newList.remove(val)
            else:
                break
        #print(self.fitnessPopulation(newPopulation))
        return newPopulation
        
            
            
            
        
    def fitnessPopulation(self,population):
        """Fitness is how many 1's,path along the wall is covered for each individual"""
        score = []
        for i in population:
            score.append(self.evaluate(i))
        #print("score is:",score)
        return score   
    def evaluate(self,indivdual):
        """Orientation of 0:North
                          1:South
                          2:East
                          3:West
        """""
        row = self.row
        col = self.col
        tempGrid = self.grid[:]
        locations =[[0,0]]
        score = 1
        orientation = 1
        for i in indivdual:
            #print(i)
            if i == 1:
                if orientation == 0:
                    orientation = 1
                elif orientation == 1:
                    orientation=2
                elif orientation == 2:
                    orientation = 3
                elif orientation == 3:
                    orientation = 0
            elif i == 2:
                if orientation == 0:
                    orientation = 3
                elif orientation == 1:
                    orientation = 0
                elif orientation == 2:
                    orientation = 1
                elif orientation == 3:
                    orientation = 2
            elif i == 3:
                if orientation == 0:
                    newLocation = [locations[-1][0],locations[-1][1]-1]
                    if newLocation[1]<0 or newLocation[1]>col: #change here to work for grid size
                        pass
                    else:
                        
                        if tempGrid[newLocation[0]][newLocation[1]] == 1 and newLocation not in locations:
                            score +=1
                        locations.append(newLocation)
                        
                elif orientation == 1:
                    newLocation = [locations[-1][0]+1,locations[-1][1]]
                    if newLocation[0]<0 or newLocation[0]>row:
                        pass
                    else:
                        if tempGrid[newLocation[0]][newLocation[1]] == 1 and newLocation not in locations:
                            score +=1
                        locations.append(newLocation)
                elif orientation == 2:
                    newLocation = [locations[-1][0],locations[-1][1]+1]
                    if newLocation[1]<0 or newLocation[1]>col: #change here to work for grid size
                        pass
                    else:
                        
                        if tempGrid[newLocation[0]][newLocation[1]] == 1 and newLocation not in locations:
                            score +=1
                        locations.append(newLocation)
                elif orientation == 3:
                    newLocation = [locations[-1][0]-1,locations[-1][1]]
                    if newLocation [0]<0 or newLocation[0]>row:
                        pass
                    else:
                        locations.append(newLocation)
                        if tempGrid[newLocation[0]][newLocation[1]] == 1 and newLocation not in locations:
                            score +=1
                        locations.append(newLocation)
        #print("Score is:",score)
        #print(locations)
        return score
    def crossover(self):
        j = -1
        newPopulation = []
        crossover_probability = 0.8
        #print(self.population)
        for i in range(len(self.population)//2):
            point = random.randint(0,self.cycles-1)
            crossover_point = random.uniform(0,1)
            if crossover_point>=crossover_probability:
                #print("cross happens")
                newPopulation.append(self.population[i][0:point][:] + self.population[j][point:-1][:])
                newPopulation.append(self.population[j][0:point][:] + self.population[i][point:-1][:])
            else:
                newPopulation.append(self.population[i])
                newPopulation.append(self.population[j])
            j-=1
        #print(newPopulation)
        #print("After cross over the fitness is:")
        #self.fitnessPopulation(newPopulation)
        #self.fitnessPopulation(newPopulation)
        def mutation(newPopulation):
            mutation_probability = 0.1
            for i in range(len(newPopulation)):
                mutation_point = random.randint(0,self.cycles-2)
                mutation_generation_prob = random.uniform(0,1)
                if mutation_probability>=mutation_generation_prob:
                    #print("Mutation Point:",mutation_point)
                    if newPopulation[i][mutation_point] == 0:
                        newPopulation[i][mutation_point] = 1
                    else:
                        newPopulation[i][mutation_point] = 0
        mutation(newPopulation)
        #print("After mutation the population is:")
        #self.fitnessPopulation(newPopulation)
ab = GA()
#ab.printPop()
#################################################################################

        