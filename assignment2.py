from search import *
from random import randint
from assignment2aux import *

def read_tiles_from_file(filename):
    # Task 1
    # Return a tile board constructed using a configuration in a file.
    # Replace the line below with your code.
    with open(filename, 'r') as file:
        lines = file.readlines()
        tiles = []
        for line in lines:
            row = []
            for char in line.rstrip('\n'):
                if char == ' ':
                    row.append(())
                elif char == 'i':
                    row.append((0,))
                elif char == 'L':
                    row.append((0, 1))
                elif char == 'I':
                    row.append((0, 2))
                elif char == 'T':
                    row.append((0, 1, 2))
            tiles.append(tuple(row))
    
    # Ensure that the number of tiles in each row is consistent
    width = len(tiles[0])
    for row in tiles:
        if len(row) != width:
            raise ValueError("Number of tiles in each row does not match")
    print(tuple(tiles))
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

    def value(self, state):
        fitness = 0
        height = len(self.tiles)
        width = len(self.tiles[0])

        for i in range(height):
            for j in range(width):
                tile = self.tiles[i][j]
                orientation = state[i * width + j]
                orientated_tile = tuple ((con + orientation) % 4 for con in tile)

                # Check connections to top, left, bottom, and right
                #if not top row
                if i > 0 and 1 in orientated_tile: 
                    #need to get the tile above it to see if it is pointing down. 
                    top_tile = self.tiles[i - 1][j]
                    top_orientation = state[(i - 1) * width + j]
                    top_orientated_tile = tuple((con + top_orientation) % 4 for con in top_tile) 
                    if 3 in top_orientated_tile:
                        fitness += 1  # Increment fitness by 1 for connection to the top
                if j > 0 and 2 in orientated_tile:
                    left_tile = self.tiles[i][j - 1]
                    left_orientation = state[i * width + j - 1]
                    left_orientated_tile = tuple((con + left_orientation) % 4 for con in left_tile)
                    if 0 in left_orientated_tile:
                        fitness += 1  # Increment fitness by 1 for connection to the left
                if i < height - 1 and 3 in orientated_tile:
                    bottom_tile = self.tiles[i + 1][j]
                    bottom_orientation = state[(i + 1) * width + j]
                    bottom_orientated_tile = tuple ((con + bottom_orientation) % 4 for con in bottom_tile)
                    if 1 in bottom_orientated_tile: 
                        fitness += 1  # Increment fitness by 1 for connection to the bottom
                if j < width - 1 and 0 in orientated_tile:
                    right_tile = self.tiles[i][j + 1]
                    right_orientation = state[i * width + j + 1]
                    right_orientated_tile = tuple((con + right_orientation) % 4 for con in right_tile)
                    if 2 in right_orientated_tile:
                        fitness += 1  # Increment fitness by 1 for connection to the right
        return fitness




# Task 3
# Configure an exponential schedule for simulated annealing.
sa_schedule = exp_schedule(k=30, lam=0.25, limit=150)
 
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
    raise NotImplementedError

def stochastic_beam_search(problem, population, limit=1000):
    # Task 6
    # Implement stochastic beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the generation limit is reached.
    # Replace the line below with your code.
    raise NotImplementedError


if __name__ == '__main__':
    # Task 1 test code
    network = KNetWalk('assignment2config.txt')
    visualise(network.tiles, network.initial)

    # Task 2 test code
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
    '''
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
    '''

    # Task 6 test code
    '''
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
    '''
