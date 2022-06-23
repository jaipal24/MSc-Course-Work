# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 14:59:15 2021

@author: reddy
"""

import random
import collections
import copy
import math

#Reference:AIMA Python Repository
#https://github.com/aimacode/aima-python
from agents import Thing, Agent, Environment, XYEnvironment, Direction, Bump, Obstacle
from search import *


# using global variables to store state,model and some locations
#______________________________________________________________________________
model = {"left": "None","right":"None","up":"None","down":"None"}
state = "not_found_home"
move_loc = []
dog_loc = [0,0]
goal_loc = []

# Things present in the environment based on the agent type
#______________________________________________________________________________
class Nut(Thing): # Nut class holds the quality of the nuts and name them good or bad based on the quality.
    def __init__(self,quality="good"):
        self.quality = quality
        if quality == "good":
            self.name = "Good_Nut"
        else:
            self.name = "Bad_Nut"

class Carrot(Thing): # This holds Carrot things in the environment
    pass

class Home(Thing): # This holds Home location of the Rat Agent
    def __init__(self,loc):
        self.loc = loc

class Ball(Thing): # Ball thing which is to be taken by dog
    pass
        
class Owner(Thing): # Owner of the dog whose location is the goal of dog
    def __init__(self,loc):
        global goal_loc
        self.loc = loc
        goal_loc = self.loc

#______________________________________________________________________________
class CuteSquirrelAgent(Agent): # This is a SIMPLE REFLEX AGENT
    location = [0,0]
    def move_next(self,loc): # this method makes the agent move from one block
        if loc == "down":
            self.location[0] += 1
        elif loc == "up":
            self.location[0] -= 1
        elif loc == "left":
            self.location[1] -= 1
        elif loc == "right":
            self.location[1] += 1
    
    def check_to_eat(self, thing): # This method will check the things in the environment, whether they are eatable by the squirrel 
        if isinstance(thing,Nut):
            if thing.quality == "good": # Squirrel eats only the good nuts and increases its performance by 10
                self.performance += 10
                return "eat"
            else: # throws the bad nuts and decreases its performance by 1
                self.performance -= 1
                return "throw"
        return "avoid"

def program_simple_reflex(percepts): #simple reflex agent program
    '''Returns an action based on it's percepts'''
    for p in percepts[0]: 
        if isinstance(p, Nut): # returns an action to eat if its nut
            return 'eat_nut'
    choice_of_dir = []
    for key,val in percepts[1].items():
        if val[1]:
            choice_of_dir.append(key)
    return random.choice(choice_of_dir) # returns a random choice of the locations that agent can move with in the ground


#______________________________________________________________________________

def update_state(model): # updates the state of the model based reflex agent and goal based agent
    global state
    for key,val in model.items():
        #print(val)
        thing = val[1][0]
        if isinstance(thing,Home): # True when the rat reaches its home and makes to move to that location
            #print(thing)
            state = ["found_home", val[0]]
            return state
        elif isinstance(thing,Owner): # True when the dog reaches its owner
            state = ["reached_owner",val[0]]
            return state
        elif isinstance(thing,Carrot): # True when the rat finds a carrot and makes to move to that location
            #print(thing)
            state = ["move_to_carrot",val[0]]
            return state
    return ["not_found_home",0]


class SneakyRat(Agent): # MODEL BASED REFLEX AGENT
    location = [0,0]
    
    def eat_carrot(self,thing): # rat will eat the carrot and increses its performance by 20
        if isinstance(thing,Carrot):
            return True
        return False

    def move_to_choosen_block(self, block_loc): # make the rat to move to next block based on the choice from percepts
        self.location = block_loc
        return True


def program_model_based_reflex(percepts): # model based reflex agent program
    global state,model,move_loc
    model = percepts[1]
    state = update_state(model)
    choice_of_path = []
    if isinstance(percepts[0],Carrot): # true when finds a carrot
        cur_act = "eat_carrot"
    else: # makes the agent decide what to do next based on the state
        cur_act = "move"
        if state[0] == "found_home":
            move_loc = state[1]
        elif state[0] == "move_to_carrot":
            move_loc = state[1]
        else:
            for key,val in percepts[1].items():
                if val[0] != "Out_of_Ground": #choose the location only in the ground available to the agent
                    if not isinstance(val[1][0],Obstacle) or val[1] == "None":
                        choice_of_path.append(val[0])
                    else:
                        choice_of_path.append(val[0])
            move_loc = random.choice(choice_of_path)
    return cur_act

#______________________________________________________________________________
class PetDog(Agent): # GOAL BASED AGENT
    location = [0,0]
    holding_the_ball = False
    def pick_the_ball(self,thing): # Dog pics the ball and hold it when true
        if isinstance(thing,Ball):
            holding_the_ball = True
            return True
        return False
    
    def move_to_choosen_block(self,loc): # makes the dog move to next block
        global dog_loc
        self.location = loc
        dog_loc = self.location
        return True
            

def program_goal_based(percepts): # goal based reflex agent program
    global state,model,move_loc,goal_loc,dog_loc
    model = percepts[1]
    state = update_state(model)
    choice_of_path = []
    if isinstance(percepts[0],Ball): # tells the dog to pick the ball
        cur_act = "pick_the_ball"
    else:
        cur_act = "move"
        if state[0] == "reached_owner": # it shows that the dog found its owner with next move
            move_loc = state[1]
        else:
            for key,val in percepts[1].items():
                if val[0] != "Out_of_Ground":
                    if not isinstance(val[1][0],Obstacle) or val[1] == "None":
                        choice_of_path.append(val[0])
                    else:
                        choice_of_path.append(val[0])
            better_dir = [0,0]
            goal = goal_loc
            for dir1 in choice_of_path:
                x = dir1[0]
                y = dir1[1]
                near_x = dog_loc[0] + better_dir[0]
                near_y = dog_loc[1] + better_dir[1]
                # https://stackoverflow.com/questions/5228383/how-do-i-find-the-distance-between-two-points#5228392
                nearest_dist = math.hypot(goal[0] - near_x,goal[1] - near_y)
                new_dist = math.hypot(goal[0] - x,goal[1] - y)
                if new_dist <= nearest_dist: # makes to choose the best move which makes the dog closer to its goal
                    better_dir = dir1
            move_loc = better_dir 
    return cur_act


#______________________________________________________________________________
class Garden(XYEnvironment): # Garden is the environment where the agent interacts
    
    def percept(self,agent): # gets the percepts of the environment from the agent
        
        if str(agent)[1:-1] == "SneakyRat" or str(agent)[1:-1] == "PetDog":
            things = [ele for ele in self.list_things_at(agent.location) if not isinstance(ele,SneakyRat)]
            x = agent.location[0]
            y = agent.location[1]
            percepts = {"left":[[x,y-1],self.list_things_at([x,y-1]) if len(self.list_things_at([x,y-1])) >0 else ["None"]] if self.is_inbounds([x,y-1]) else ["Out_of_Ground",[0]],
                    "right":[[x,y+1], self.list_things_at([x,y+1]) if len(self.list_things_at([x,y+1])) >0 else ["None"]] if self.is_inbounds([x,y+1]) else ["Out_of_Ground",[0]],
                    "up":[[x-1,y], self.list_things_at([x-1,y]) if len(self.list_things_at([x-1,y])) >0 else ["None"]] if self.is_inbounds([x-1,y]) else ["Out_of_Ground",[0]],
                    "down":[[x+1,y], self.list_things_at([x+1,y]) if len(self.list_things_at([x+1,y])) >0 else ["None"]] if self.is_inbounds([x+1,y]) else ["Out_of_Ground",[0]]
                    }
            return [things[0] if len(things) > 0 else "No_Things",percepts]
        elif str(agent)[1:-1] == "CuteSquirrelAgent":
            things = self.list_things_at(agent.location)
            x = agent.location[0]
            y = agent.location[1]
            percepts = {"left":[[x,y-1],self.is_inbounds([x,y-1])],
                    "right":[[x,y+1],self.is_inbounds([x,y+1])], 
                    "up":[[x-1,y],self.is_inbounds([x-1,y])], 
                    "down":[[x+1,y],self.is_inbounds([x+1,y])]
                    }
            return [things,percepts]
        
    
    
    def execute_action(self,agent,action): # performs required actions on the environment based on the received percepts
        global move_loc
        
        if action == 'right':
            print('{} decided to move {} at location: {} with performance: {}'.format(str(agent)[1:-1], action, agent.location, agent.performance))
            agent.move_next(action)
        elif action == 'left':
            print('{} decided to move {} at location: {} with performance: {}'.format(str(agent)[1:-1], action, agent.location, agent.performance))
            agent.move_next(action)
        elif action == 'up':
            print('{} decided to move {} at location: {} with performance: {}'.format(str(agent)[1:-1], action, agent.location, agent.performance))
            agent.move_next(action)
        elif action == 'down':
            print('{} decided to move {} at location: {} with performance: {}'.format(str(agent)[1:-1], action, agent.location, agent.performance))
            agent.move_next(action)
        elif action == "eat_nut":
            items = self.list_things_at(agent.location, tclass=Nut)
            if len(items) != 0:
                if agent.check_to_eat(items[0]) == "eat":
                    print('{} ate {} at location: {} with performance: {}'
                          .format(str(agent)[1:-1], items[0].name, agent.location,agent.performance))
                    self.delete_thing(items[0])
                elif agent.check_to_eat(items[0]) == "throw":
                    print('{} throws {} at location: {} with performance: {}'
                          .format(str(agent)[1:-1], items[0].name, agent.location,agent.performance))
                    self.delete_thing(items[0])
                else:
                    print('{} avoids {} at location: {} with performance: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location,agent.performance))
        elif action == "eat_carrot":
            items = self.list_things_at(agent.location, tclass=Carrot)
            #print(items)
            if len(items) != 0:
                if agent.eat_carrot(items[0]):
                    print('{} ate {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "pick_the_ball":
            items = self.list_things_at(agent.location, tclass=Ball)
            #print(items)
            if len(items) != 0:
                if agent.pick_the_ball(items[0]):
                    print('{} picked the {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])
        elif action == "move":
            #print("trying to move")
            if agent.move_to_choosen_block(move_loc):
                print('{} moved to location: {}'.format(str(agent)[1:-1],move_loc))
            
            
        
    
    def is_done(self): # when the agent finds nothing to eat in the environment the agent stops or if the agent reaches its home or its owner
        for agent in self.agents:
            if str(agent)[1:-1] == "SneakyRat":
                for thing in self.things:
                    if isinstance(thing,Home):
                        if thing.loc == agent.location:
                            print("SneakyRat reached Home at ",thing.loc)
                            return True
            elif str(agent)[1:-1] == "PetDog":
                for thing in self.things:
                    if isinstance(thing,Owner):
                        if thing.loc == agent.location:
                            print("Pet reached Owner at ",thing.loc)
                            if agent.holding_the_ball:
                                print("The dog was able to pick the ball on the way to owner")
                            else:
                                print("The dog didn't pick the ball on the way to owner")
                            return True
            elif str(agent)[1:-1] == "CuteSquirrelAgent":
                no_edibles = not any(isinstance(thing, Nut) for thing in self.things)
                return no_edibles
        return False

#______________________________________________________________________________
# reference from search.py

# creating a map for all the tourist plases in paris
Paris_Tour_Guide_Map = UndirectedGraph(dict(
    Eiffel_Tower=dict(Norte_Dame=153, Holy_Chapel=124, River_Cruise=79),
    Norte_Dame=dict(Tuileries_Garden=132, Orsay_Museum=70),
    Alexander_Bridge=dict(Champs_Elysees=100, River_Cruise=183),
    Patheon=dict(Eiffel_Tower=83, Rodin_Museum=37),
    Royal_Palace=dict(Champs_Elysees=214, Louvre_Museum=179),
    Arc_De_Triamphe=dict(Orsay_Museum=92, Patheon=105),
    Rodin_Museum=dict(Eiffel_Tower=81, Sacre_Coeur_Basilica=186),
    Luxen_Bourg_Garden=dict(Arc_De_Triamphe=108),  
    ))



Eiffel_Tower_to_Royal_Palace = GraphProblem('Eiffel_Tower', 'Royal_Palace', Paris_Tour_Guide_Map)
Holy_Chapel_to_River_Cruise = GraphProblem('Holy_Chapel', 'River_Cruise', Paris_Tour_Guide_Map)
Rodin_Museum_to_Alexander_Bridge = GraphProblem('Rodin_Museum', 'Alexander_Bridge', Paris_Tour_Guide_Map)


def compare_searchers(problems, header,
                      searchers=[breadth_first_tree_search,
                                 breadth_first_graph_search,
                                 depth_first_graph_search,
                                 iterative_deepening_search,
                                 depth_limited_search,
                                 recursive_best_first_search]):
    def do(searcher, problem):
        p = InstrumentedProblem(problem)
        if searcher == best_first_graph_search:
            searcher(p,problem.h)
        else:
            searcher(p)
        return p

    table = [[name(s)] + [do(s, p) for p in problems] for s in searchers]
    print_table(table, header)



def compare_graph_searchers(searchers):
    """Prints a table of search results."""    
    outputString = '\nActions/Goal Tests/States/Goal\n'    
    print(outputString)    
    compare_searchers(problems=[Eiffel_Tower_to_Royal_Palace, Holy_Chapel_to_River_Cruise, Rodin_Museum_to_Alexander_Bridge], 
                      header=['Searcher', 'Paris_Map(Eiffel_Tower, Royal_Palace)',
                              'Paris_Map(Holy_Chapel, River_Cruise)', 
                              'Paris_Map(Rodin_Museum, Alexander_Bridge)'], 
                      searchers=searchers)


uninformed_searchers=[breadth_first_graph_search,depth_first_graph_search,iterative_deepening_search]
compare_graph_searchers(uninformed_searchers)

informed_searchers=[best_first_graph_search,uniform_cost_search,astar_search]
compare_graph_searchers(informed_searchers)


#______________________________________________________________________________    
def query1(): # execution of simple reflex agent
    garden = Garden(3,3)
    squirrel = CuteSquirrelAgent(program_simple_reflex)
    nut1 = Nut()
    nut2 = Nut()
    carrot = Carrot()
    garden.add_thing(squirrel,[0,0])
    garden.add_thing(nut1,[0,1])
    garden.add_thing(nut2,[1,0])
    garden.add_thing(carrot,[1,1])
    garden.run(10)


def query2(): # execution of model based reflex agent
    garden = Garden(3,3)
    rat = SneakyRat(program_model_based_reflex)
    carrot = Carrot()
    obst = Obstacle()
    home = Home([2,2])
    garden.add_thing(rat,[0,0])
    garden.add_thing(carrot,[2,0])
    garden.add_thing(obst,[1,1])
    garden.add_thing(home,home.loc)
    garden.run(10)
    
def query3(): # execution of goal based agent
    garden = Garden(5,5)
    dog = PetDog(program_goal_based)
    ball = Ball()
    obst = Obstacle()
    owner = Owner([4,4])
    garden.add_thing(dog,[0,0])
    garden.add_thing(ball,[2,0])
    garden.add_thing(obst,[1,1])
    garden.add_thing(owner,owner.loc)
    garden.run(10)

print("\nExecuting Simple Reflex Agent\n")
query1()
print("\nExecuting Model Based Reflex Agent\n")
query2()
print("\nExecuting Goal Based Agent\n")
query3()

