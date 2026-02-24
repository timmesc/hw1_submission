"""
gen_test_json.py — Generate N random 101x101 mazes and save as mazes.json. Uses same algorithm as maze_generator.py.

Usage:
    python gen_test_json.py [--num_mazes N] [--seed S] [--output FILE]
"""
import json
import random
import argparse
from constants import ROWS, START_NODE, END_NODE
from tqdm import tqdm




# set random seed for reproducibility
random.seed(42)

def create_maze() -> list:
    # TODO: Implement this function to generate and return a random maze as a 2D list of 0s and 1s.

    size = ROWS
    grid = [[0] * size for n in range(size)]
    visited = [[False] * size for n in range(size)]


    def get_neighbors(row,col):
        neighbors = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
         nr, nc = row + dr, col + dc 
         if 0 <= nr < size and 0 <= nc < size:
                neighbors.append((nr,nc))
        return neighbors

    #start DFS from a random cell 
    start_r = random.randint(0,size-1)
    start_c = random.randint(0,size-1)

    visited[start_r][start_c] = True
    grid[start_r][start_c] = 0
    stack = [(start_r,start_c)]

    unblocked_blocked = [0,1]
    weights = [0.7, 0.3]

    max_iterations = size * size * 100
    iteration = 0 
    current_row, current_col = start_r, start_c

    while not all(visited[r][c] for r in range(size) for c in range(size)) and iteration < max_iterations:
        iteration +=1
        neighbors = get_neighbors(current_row, current_col)
        unvisited = [(r,c) for r,c in neighbors if not visited[r][c]]

        if unvisited:
                current_row, current_col = random.choice(unvisited)
                visited[current_row][current_col] = True 
                grid[current_row][current_col] = random.choices(unblocked_blocked,weights, k=1)[0]  
                if grid[current_row][current_col] == 0:
                    stack.append((current_row,current_col))

        else: 
            
            found_valid_cell = False
            while stack: 
                current_row, current_col = stack.pop()
                neighbors = get_neighbors(current_row,current_col)
                unvisited = [(r,c) for r,c in neighbors if not visited[r][c]]

                if unvisited: 
                    found_valid_cell = True 
                    break
                   
            if not found_valid_cell: 
                unvisited_position = [(r,c) for r in range (size) for c in range(size) if not visited[r][c]]
                if unvisited_position:
                    current_row, current_col = unvisited_position[0]
                    visited[current_row][current_col] = True
                    grid[current_row][current_col] = 0
                    stack.append((current_row,current_col))

    grid[START_NODE[0]][START_NODE[1]] = 0
    grid[END_NODE[0]][END_NODE[1]] = 0
    return grid



def main():
    parser = argparse.ArgumentParser(description="Generate random mazes as JSON")
    parser.add_argument("--num_mazes", type=int, default=50,
                        help="Number of mazes to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="mazes.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    random.seed(args.seed)
    
    mazes = []
    for _ in tqdm(range(args.num_mazes), desc="Generating mazes"):  
        mazes.append(create_maze())

    with open(args.output, "w") as fp:
        json.dump(mazes, fp)
    print(f"Generated {args.num_mazes} mazes (seed={args.seed}) -> {args.output}")

if __name__ == "__main__":
    main()
