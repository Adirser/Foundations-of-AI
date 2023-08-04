import copy
import heapq
import math
import random

class board:
    def __init__(self, current_board=[[], [], [], [], [], []], goal_board=[[], [], [], [], [], []]):
        self.current_board = current_board
        self.goal_board = goal_board
        self.parent = None
        self.g = 0
        self.h = 0
        self.f = 0
    def __lt__(self, other):
        return self.f < other.f
    def calculate_heuristic(self):
        agent_locations = self.agents_locations()
        agent_goals = self.agents_goals()
        extra_agents = self.findErrors2(agent_locations,agent_goals)
        distance = 0
        for i in agent_goals:
            agent_location_index = 0
            min_dist = 10000
            for j in agent_locations:
                if abs(j[0] - i[0]) + abs(j[1] - i[1]) < min_dist:
                    min_dist =  abs(j[0] - i[0]) + abs(j[1] - i[1])
                    index = agent_location_index
                agent_location_index+=1
            distance += min_dist
            try:
                agent_locations.pop(index)
            except:
                print()
        self.h = distance + extra_agents
        return self.h
    def agents_locations(self):
        agents_locations = []
        row = 0
        col = 0
        for i in self.current_board:
            for j in i:
                if j==2:
                    agents_locations.append([row,col])
                col +=1
            row+=1
            col=0
        return agents_locations
    def agents_goals(self):
        agents_goals = []
        row = 0
        col = 0
        for i in self.goal_board:
            for j in i:
                if j == 2:
                    agents_goals.append([row,col])
                col+=1
            row+=1
            col=0
        return agents_goals
    def calc_f(self):
        self.f = self.calculate_heuristic() + 1
        return self.f
    def findErrors2(self,agent_locations,agent_goals):  # Finds errors of type 3 (Agents that need to exit the board)
        num_of_extra_agents = len(agent_locations)-len(agent_goals)
        distance = 0
        distanceFromBottom = []
        for i in range(num_of_extra_agents):
            for i in agent_locations:
                distanceFromBottom.append(abs(6-i[0]))
            distanceFromBottom.sort()
            distance += distanceFromBottom[0]
        return distance
    def print_board(self):  # prints the board
        index = 1
        print(" ", end=" ")
        for i in range(1, 7):
            print("", i, end="")
        print()
        for i in self.current_board:
            print(index, end="):")
            for j in i:
                if j == 2:
                    print("*", "", end="")
                if j == 1:
                    print("@", "", end="")
                if j == 0:
                    print(" ", "", end="")

            print()
            index += 1
    def check_potential_moves(self):  # Checks the potential moves the current board has and then calculates the costs per board
        moves_and_boards = []
        agent_locations = []
        moves = []
        costs_and_boards = []
        indexR = 0
        for i in self.current_board:
            indexC = 0
            for j in i:
                if (j == 2):
                    agent_locations.append([indexR, indexC])
                indexC += 1
            indexR += 1
        for i in agent_locations:
            if i[1] + 1 <= 5:
                if self.current_board[i[0]][i[1] + 1] == 0:
                    temp = copy.deepcopy(self)
                    moves.append([i[0], i[1] + 1])
                    temp.current_board[i[0]][i[1] + 1] = 2
                    temp.current_board[i[0]][i[1]] = 0
                    costs_and_boards.append([temp.calc_f(), temp])
                    moves_and_boards.append([i[0], i[1] + 1,temp])
            if i[1] - 1 >= 0:
                if self.current_board[i[0]][i[1] - 1] == 0:
                    temp = copy.deepcopy(self)
                    moves.append([i[0], i[1] - 1])
                    temp.current_board[i[0]][i[1] - 1] = 2
                    temp.current_board[i[0]][i[1]] = 0
                    costs_and_boards.append([temp.calc_f(), temp])
                    moves_and_boards.append([i[0], i[1] - 1,temp])


            if i[0] - 1 >= 0:
                if self.current_board[i[0] - 1][i[1]] == 0:
                    temp = copy.deepcopy(self)
                    moves.append([i[0] - 1, i[1]])
                    temp.current_board[i[0] - 1][i[1]] = 2
                    temp.current_board[i[0]][i[1]] = 0
                    costs_and_boards.append([temp.calc_f(), temp])
                    moves_and_boards.append([i[0]-1, i[1],temp])

            if i[0] + 1 <= 6:
                if i[0] + 1 == 6:
                    temp = copy.deepcopy(self)
                    moves.append([i[0] + 1, i[1]])
                    temp.current_board[i[0]][i[1]] = 0
                    costs_and_boards.append([temp.calc_f(), temp])
                    moves_and_boards.append([i[0]+1, i[1],temp])

                elif self.current_board[i[0] + 1][i[1]] == 0:
                    temp = copy.deepcopy(self)
                    moves.append([i[0] + 1, i[1]])
                    temp.current_board[i[0] + 1][i[1]] = 2
                    temp.current_board[i[0]][i[1]] = 0
                    costs_and_boards.append([temp.calc_f(), temp])
                    moves_and_boards.append([i[0]+1, i[1],temp])
        return costs_and_boards
def check_identical(board_1, board_2):  # Checks if 2 boards are identical
    flag = False
    if board_1 == board_2:
        flag = True
    return flag
def check_exists(closed_list, potential_board):  # Checks if a path in the open list is already in the closed list
    for i in closed_list:
        if check_identical(i[1].current_board, potential_board[1].current_board) == True:
            return True
    return False
def finish(boardz2):
    trace = []
    while (boardz2.parent != None):
        a = copy.deepcopy(boardz2)
        trace.append(a)
        boardz2 = copy.deepcopy(boardz2.parent)
    index =1
    for i in  reversed(trace):
        print("Board number " , index )
        i.print_board()
        print("Heuristic:<", i.h, ">")
        index+=1
def a_star_seach(starting_board,goal_board,detail_output):
    boardz = board(starting_board, goal_board)
    print("Starting Board:")
    boardz.print_board()
    print("Heuristic is : " , boardz.calc_f()-1)
    print("-------------------------------")
    open_list = []
    heapq.heapify(open_list)
    heapq.heappush(open_list, (boardz.calc_f(), boardz))
    closed_List = []
    index = 1
    counter = 0
    while (len(open_list) > 0 and check_identical(boardz.current_board, goal_board) == False):
        counter += 1
        potential_boards_for_current_board = boardz.check_potential_moves()  # retrieving all potential boards for best board
        for i in potential_boards_for_current_board:  # Inserting to open list all potential board except those that are inside closed list
            i[1].parent = copy.deepcopy(boardz)
            if check_exists(closed_List, i) == False:
                heapq.heappush(open_list, (i[0], i[1]))
        x = heapq.heappop(open_list)
        boardz = copy.deepcopy(x[1])
        boardz.calc_f()
        closed_List.append(x)  # Takes the lowest cost board

        if detail_output == False:
            print("Board number ", index)
            boardz.print_board()
            print("Heuristic:<", boardz.h, ">")
            print("--------")
            index += 1
        if (len(open_list) == 0 or counter > 1000):
            print("No path found.")
            exit()
    if detail_output == True:
        finish(boardz)
def hill_climbing_search(starting_board,goal_board,detail_output):
    game_board = board(starting_board,goal_board)
    boards = []
    heapq.heapify(boards)
    first_game_board_neighbors = game_board.check_potential_moves()
    restart_counter = 0
    while check_identical(game_board.current_board, goal_board) == False and restart_counter < 5:
        x = game_board.check_potential_moves()
        for i in x : # Generate Potential moves and inserting them into a heap
            i[1].parent = game_board
            if game_board.calc_f() > i[0]:
                heapq.heappush(boards,(i[0],i[1]))
        if len(boards) == 0:
            restart_counter+=1
            x = random.choice(first_game_board_neighbors)
            first_game_board_neighbors.remove(x)
            game_board = x[1]
            print("R-E-S-T-A-R-T-I-N-G ......")
        else:
            x = heapq.heappop(boards)
            game_board = x[1]
            if detail_output == False:
                game_board.print_board()
                print("Heuristic < " ,game_board.h," >")
            boards = []
            heapq.heapify(boards)
    finish(game_board)
def simulated_annealing_search(starting_board,goal_board,detail_output):
    current_board = board(starting_board,goal_board)
    if current_board.calc_f() <= 100: # Limiting temperature to 100
        temperature = current_board.calc_f()
    else:
        temperature = 100
    print("Starting Board :")
    current_board.print_board()
    while check_identical(current_board.current_board,goal_board) == False and temperature>0:
        y = current_board.check_potential_moves() # Generates potential Boards
        x = random.choice(y) # Chooses a random board
        x[1].parent = current_board
        temperature -= 0.05 # Decreasing temperature by 0.05 point each iteration
        a = x[1].calc_f() # Canidate Heuristic
        b = current_board.calc_f() #Current Heuristic
        delta_E = b-a #Current - Canidate
        if delta_E > 0:
                current_board = copy.deepcopy(x[1])
                if detail_output == False:
                    print("Probability is : 1 ")
                    current_board.print_board()
        else:
            if temperature > 0:
                probability = math.exp(delta_E/temperature)
                rand_num = random.random()
                if rand_num <= probability:
                    current_board = x[1]
                    if detail_output == False:
                        print("Probability is : ", probability)
                        current_board.print_board()
    if detail_output == True:
        finish(current_board)
def k_beam_search(starting_board,goal_board,detail_output):
    open_list = []
    closed_list = []
    predecessors = []
    heapq.heapify(open_list)
    game_board = board(starting_board, goal_board)
    heapq.heappush(open_list,(game_board.calc_f(),game_board))
    predecessors.append([game_board.calc_f(),game_board])
    while len(open_list) > 0 and check_identical(game_board.current_board,goal_board) == False:
        x = game_board.check_potential_moves()
        temp_list = []
        heapq.heapify(temp_list)
        for i in x:
            if check_exists(closed_list,i) == False:
                i[1].parent = copy.deepcopy(game_board)
                heapq.heappush(temp_list,i)
        for i in range(3): # Pushing 3 Best boards into open list
            y = heapq.heappop(temp_list)
            heapq.heappush(open_list,(y[0],y[1]))
            predecessors.append(y)
        z = heapq.heappop(open_list) # Choosing the best 1 out of the entire list of solutions
        closed_list.append(z)
        game_board = z[1]
        predecessors.append([game_board.calc_f(),game_board])
    game_board.print_board()
    counter = 0
    if detail_output == True:
        finish(game_board)
    if detail_output == False:
        for i in predecessors:
            if counter%4 == 0:
                print("Starting Board :")
                i[1].print_board()
            if counter%4 == 1:
                print("Board a")
                i[1].print_board()
            if counter%4 == 2:
                print("Board b:")
                i[1].print_board()
            if counter%4 == 3:
                print("Board c:")
                i[1].print_board()
            counter+=1
def check_population(population,goal_board,detail_output): #Checks if out of the entire population any solution is the goal state
    for i in population:
        if check_identical(i[1].current_board,goal_board) == True:
            print("----------------")
            i[1].print_board()
            print("^^^ Fittest Board Found ^^^")
            print("----------------")

            if detail_output == True:
                find_path(recreating_path,goal_board,1,True)

            return True
    return False
def merge(board_1,board_2,prob_1,prob_2,detail_output):
    crosspoint = int(random.random()*6)
    boardy = []
    if detail_output == False:
        print("Starting Board 1 : probability of selection from population ::<" ,prob_1, ">")
        board_1[1].print_board()
        print("_____")
        print("Starting Board 2 : probability of selection from population ::<" ,prob_2, ">")
        board_2[1].print_board()
        print("_____")
    for i in range(6):
        if i >= crosspoint:
            x = copy.deepcopy(board_2[1].current_board[i])
            boardy.append(x)
        else:
            x = copy.deepcopy(board_1[1].current_board[i])
            boardy.append(x)
    new_board = board(boardy,board_1[1].goal_board)
    new_board.parent = copy.deepcopy(board_1[1])
    rand = random.random()
    flag = False
    if rand <= 0.5: #Mutations are great and after a little bit of senstivity analysis 0.5 got the best result
        flag = True
        for i in range(random.randint(3,10)): #Mutation is Happening in 3-10 cells (as long as the cell is different than a forcefield)
            rand_1 = random.randint(0, 5)
            rand_2 = random.randint(0, 5)
            if new_board.current_board[rand_1][rand_2] != 1:
                new_board.current_board[rand_1][rand_2] = random.choice([0,2])
    child = [new_board.calc_f(),new_board]
    if detail_output == False:
        if flag == True:
            print("Result Board : Mutation happend ::<YES>")
        else:
            print("Result Board : Mutation happend ::<NO>")
        new_board.print_board()
    return child
def create_decendents(population,detail_output):
    new_population = [] # New population
    temp_population = copy.deepcopy(population)
    heapq.heapify(temp_population)
    # Implementation of the Elitesm Concept
    num_of_elites = int(random.random()*2)+1
    for i in range(num_of_elites):
        new_population.append(heapq.heappop(temp_population))
    totalHeuristic = 0
    probabilities = []
    counter = 0
    for i in population: #Updating Heuristics and Calculating total Heuristics for the fitness function
        i[0] = i[1].calc_f()
        totalHeuristic += i[0]
    for i in population: # Calculating probabilities
        probability_for_child = (((totalHeuristic)-i[0])/totalHeuristic)/(len(population)-1)
        ranges = counter
        counter += probability_for_child
        probabilities.append([ranges, counter])
    counter = 0
    for k in range(10-num_of_elites): #Creating Offsprings , Selection Happends based on probability
        random_number_1 = random.random()
        random_number_2 = random.random()
        chosen = []
        for i in probabilities:
            if  random_number_1 > i[0] and  random_number_1 <= i[1]:
                chosen.append([population[counter][0], population[counter][1]])
                prob_1 = i[1] - i[0]
            counter += 1
        counter = 0
        for i in probabilities:
            if  random_number_2 > i[0] and  random_number_2 < i[1]:
                chosen.append([population[counter][0], population[counter][1]])
                prob_2 = i[1] - i[0]
            counter += 1
        counter = 0
        new_population.append(merge(chosen[0],chosen[1],prob_1,prob_2,detail_output))
    return new_population
def genetic_search(starting_board , goal_board,detail_output):
    current_board = board(starting_board,goal_board)
    global recreating_path
    recreating_path = starting_board
    population = []
    x = current_board.check_potential_moves()
    if len(x)>10:
        for i in range(10): #Choosing 10 Randon Boards as the initial population
            y = random.choice(x)
            population.append(y)
            x.remove(y)
    else:
        for i in range(len(x)):
            y = random.choice(x)
            population.append(y)
            x.remove(y)
    generations =0
    while generations < 200 and check_population(population,goal_board,detail_output) == False:
        population = create_decendents(population,detail_output)
        generations+=1

def find_path(starting_board, goal_board, search_method, detail_output):
    if search_method == 1:
        a_star_seach(starting_board,goal_board,detail_output)
    if search_method == 2:
        hill_climbing_search(starting_board,goal_board,detail_output)
    if search_method == 3:
        simulated_annealing_search(starting_board,goal_board,detail_output)
    if search_method == 4:
        k_beam_search(starting_board,goal_board,detail_output)
    if search_method == 5 :
        genetic_search(starting_board,goal_board,detail_output)

