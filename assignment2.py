from search import *
from random import randint
from assignment2aux import *

def read_tiles_from_file(filename):
    # Task 1
    # Return a tile board constructed using a configuration in a file.
    # Replace the line below with your code.
    #opens the file in read 
    with open(filename, 'r') as file:
        lines = file.readlines()
        tiles = []
        #loops for each character in each line of the file, other than the new line char. 
        for line in lines:
            row = []
            for char in line.rstrip('\n'):
                #blank tile 
                if char == ' ':
                    row.append(())
                #i tile 
                elif char == 'i':
                    row.append((0,))
                #L tile
                elif char == 'L':
                    row.append((0, 1))
                #I tile
                elif char == 'I':
                    row.append((0, 2))
                #T tile
                elif char == 'T':
                    row.append((0, 1, 2))
            #adds each row to tiles list 
            tiles.append(tuple(row))
    
    #checks that the number of tiles in each row is consistent
    width = len(tiles[0])
    for row in tiles:
        if len(row) != width:
            raise ValueError("Number of tiles in each row does not match")
    #returns tiles as a tuple
    return tuple(tiles)


class KNetWalk(Problem):
    def __init__(self, tiles):
        if type(tiles) is str:
            self.tiles = read_tiles_from_file(tiles)
        else:
            self.tiles = tiles
        height = len(self.tiles)
        width = len(self.tiles[0])
        self.max_fitness = sum(sum(len(tile) for tile in row) for row in self.tiles)
        super().__init__(self.generate_random_state())

    def generate_random_state(self):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [randint(0, 3) for _ in range(height) for _ in range(width)]

    def actions(self, state):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [(i, j, k) for i in range(height) for j in range(width) for k in [0, 1, 2, 3] if state[i * width + j] != k]

    def result(self, state, action):
        pos = action[0] * len(self.tiles[0]) + action[1]
        return state[:pos] + [action[2]] + state[pos + 1:]

    def goal_test(self, state):
        return self.value(state) == self.max_fitness

    #calculates the fitness of each state
    def value(self, state):
        fitness = 0
        height = len(self.tiles)
        width = len(self.tiles[0])

        #loops through every tile 
        for i in range(height):
            for j in range(width):
                #sets the current tile to tile
                tile = self.tiles[i][j]
                #gets the orientation of the tile
                orientation = state[i * width + j]
                #orientates the tile and saves as a tuple
                orientated_tile = tuple ((con + orientation) % 4 for con in tile)

                # Check connections to top, left, bottom, and right
                #if not top row and pointing up
                if i > 0 and 1 in orientated_tile: 
                    #gets the tile above and orientates the the correct way
                    top_tile = self.tiles[i - 1][j]
                    top_orientation = state[(i - 1) * width + j]
                    top_orientated_tile = tuple((con + top_orientation) % 4 for con in top_tile) 
                    #checks if the tile above is pointing down 
                    if 3 in top_orientated_tile:
                        #adds one to the fitness
                        fitness += 1  
                #if not left column and pointing left 
                if j > 0 and 2 in orientated_tile:
                    #get the tile to the left and orientes it the correct way
                    left_tile = self.tiles[i][j - 1]
                    left_orientation = state[i * width + j - 1]
                    left_orientated_tile = tuple((con + left_orientation) % 4 for con in left_tile)
                    #checks if the tile to the left is pointing right
                    if 0 in left_orientated_tile:
                        #adds one to the fitness
                        fitness += 1  
                #if not bottom row and is pointing down
                if i < height - 1 and 3 in orientated_tile:
                    #gets the tile under it and orientates it the correct way
                    bottom_tile = self.tiles[i + 1][j]
                    bottom_orientation = state[(i + 1) * width + j]
                    bottom_orientated_tile = tuple ((con + bottom_orientation) % 4 for con in bottom_tile)
                    #if the tile under it is poinitng up
                    if 1 in bottom_orientated_tile: 
                        #adds one to the fitness
                        fitness += 1  
                #if not right column and is pointing right 
                if j < width - 1 and 0 in orientated_tile:
                    #gets the tile to the right and orientates it the correct way
                    right_tile = self.tiles[i][j + 1]
                    right_orientation = state[i * width + j + 1]
                    right_orientated_tile = tuple((con + right_orientation) % 4 for con in right_tile)
                    #if the tile to the right is pointing left
                    if 2 in right_orientated_tile:
                        #adds one to the fitness
                        fitness += 1  
        #returns the total fitness of the state
        return fitness




# Task 3
# Configure an exponential schedule for simulated annealing.
sa_schedule = exp_schedule(k=32, lam=0.25, limit=150)
 
# Task 4
# Configure parameters for the genetic algorithm.
pop_size = 55
num_gen = 120
mutation_prob = 0.3

def local_beam_search(problem, population):
    # Task 5
    # Implement local beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the next population contains no fitter state.
    # Replace the line below with your code.
    
    # Get the beam width from the initial population size
    beam_width = len(population)
    
    # stores the fittest state in the previous population
    fittest_prev_gen = max(population, key=problem.value)
    
    #loop until a goal state is found or termination conditions are met
    while True:
        #generate child states for each state in the current population
        children = []
        for state in population:
            for action in problem.actions(state):               
                child = problem.result(state, action)
                children.append(child)
        
        #sort the child states by their fitness in descending order
        sorted_children = sorted(children, key=problem.value, reverse=True)
        
        #keep the top b amount of fittest child states as the new population
        population = sorted_children[:beam_width]
        
        #check if goal state is found in the population
        for state in population:
            if problem.goal_test(state):
                return state
        
        #TERMINATION CHECK
        #storess the current fittest to compare
        fittest_current_gen = max(population, key=problem.value)
        #checks if the fittest state is more than the last one else returns the prevous gen
        if problem.value(fittest_current_gen) <= problem.value(fittest_prev_gen):
            return fittest_prev_gen
        
        # update the curr gen to the old gen
        fittest_prev_gen = fittest_current_gen
      


def stochastic_beam_search(problem, population, limit=1000):
    # Task 6
    # Implement stochastic beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the generation limit is reached.
    # Replace the line below with your code.
   
    # Initialize the current population
    current_population = population
    
    for _ in range(limit):
        # Expand each state in the current population
        child_states = []
        for state in current_population:
            for action in problem.actions(state):
                child = problem.result(state, action)
                child_states.append(child)
        
        #does fitness-weighted random sampling to select the next population
        # turns child_states into nodes 
        child_nodes = [Node(child) for child in child_states]
        #calulates fitness for each child state
        fitness_values = [problem.value(child.state) for child in child_nodes]
        #calculate probabilities for sampling
        probabilities = [value / sum(fitness_values) for value in fitness_values]
        #perform weighted random sampling without replacement
        sampled_nodes = np.random.choice(child_nodes, len(current_population), replace=False, p=probabilities)
        #convert sampled Nodes back to states
        next_population = [node.state for node in sampled_nodes]
        
        #if a goal state is found return that state
        for state in next_population:
            if problem.goal_test(state):
                return state
        
        #update the current population
        current_population = next_population
    
    #return the fittest state in the current population if no goal state is found
    fittest_state = max(current_population, key=lambda state: problem.value(state))
    return fittest_state


if __name__ == '__main__':
    # Task 1 test code
    network = KNetWalk('assignment2config.txt')
    visualise(network.tiles, network.initial)

    #Task 2 test code
    run = 0
    method = 'hill climbing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = hill_climbing(network)
        print(f'{method} run {run}: state fitness {network.value(state)} out of {network.max_fitness}')
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)

    

    # Task 3 test code
    
    run = 0
    method = 'simulated annealing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = simulated_annealing(network, schedule=sa_schedule)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 4 test code
    
    run = 0
    method = 'genetic algorithm'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = genetic_algorithm([network.generate_random_state() for _ in range(pop_size)], network.value, [0, 1, 2, 3], network.max_fitness, num_gen, mutation_prob)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 5 test code
    
    run = 0
    method = 'local beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = local_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 6 test code
    
    run = 0
    method = 'stochastic beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = stochastic_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    