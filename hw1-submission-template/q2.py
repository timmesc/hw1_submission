"""
q2.py — Repeated Forward A* (Forward Replanning) with tie-breaking variants + Pygame visualization

Renders TWO views side-by-side:
- LEFT  : full (ground-truth) maze used for the run
- RIGHT : agent knowledge + search visualization

Controls:
- R : generate a new random maze and run again (max-g by default)
- 1 : run MAX-G on the current maze
- 2 : run MIN-G on the current maze
- L : load maze from text file (see readFile format) and run MAX-G
- ESC or close window : quit

Maze file format (readFile):
- Space-separated 0/1 values, 1 = blocked, 0 = free, one row per line.

Legend (colors):
GREY   = expanded / frontier / unknown (unseen)
PATH   = executed path
YELLOW = start + agent position
BLUE   = goal
WHITE  = known free
BLACK  = known blocked
"""

from __future__ import annotations

import heapq
import argparse
import json
import time
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import pygame
from constants import ROWS, START_NODE, END_NODE, BLACK, WHITE, GREY, YELLOW, BLUE, PATH, NODE_LENGTH, GRID_LENGTH, WINDOW_W, WINDOW_H, GAP
from custom_pq import CustomPQ_maxG, CustomPQ_minG

def readMazes(fname: str) -> List[List[List[int]]]:
    """
    Reads a JSON file containing a list of mazes.
    Each maze is a list of ROWS lists, each with ROWS int values (0=free, 1=blocked).
    Returns a list of maze[r][c] grids.
    """
    with open(fname, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    mazes: List[List[List[int]]] = []
    for idx, grid in enumerate(data):
        if len(grid) != ROWS or any(len(row) != ROWS for row in grid):
            raise ValueError(f"Maze {idx}: expected {ROWS}x{ROWS}, got {len(grid)}x{len(grid[0]) if grid else 0}")
        maze = [[int(v) for v in row] for row in grid]
        maze[START_NODE[0]][START_NODE[1]] = 0
        maze[END_NODE[0]][END_NODE[1]] = 0
        mazes.append(maze)
    return mazes


# ---- helper functions (also used by q3 and q5) ----

def manhattan_distance(cell, goal):
    """Manhattan distance heuristic"""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def get_neighbors(cell, size):
    """Return valid neighboring cells (N, S, W, E)"""
    row, col = cell
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size:
            neighbors.append((nr, nc))
    return neighbors


def a_star_search(known_grid, start, goal, size, tie_breaking, counter, g, search_val,
                  h_values=None, return_closed=False):
    """
    Single A* search on agent's known grid.
    Returns (path, expanded) or (path, expanded, closed_list) if return_closed=True.
    """
    g[start] = 0
    search_val[start] = counter
    g[goal] = float('inf')
    search_val[goal] = counter

    tree = {}
    closed_set = set()
    closed_list = []

    # pick the right pq for tie-breaking
    if tie_breaking == "max_g":
        pq = CustomPQ_maxG()
    else:
        pq = CustomPQ_minG()

    def h_of(cell):
        # use adaptive heuristic if available, otherwise manhattan
        if h_values is not None and cell in h_values:
            return h_values[cell]
        return manhattan_distance(cell, goal)

    h = h_of(start)
    pq.push(h, 0, start)

    expanded = 0

    while not pq.is_empty():
        f_val, g_val, s = pq.pop()

        if s in closed_set:
            continue

        # goal found, dont count as expanded
        if s == goal:
            break

        closed_set.add(s)
        closed_list.append(s)
        expanded += 1

        for neighbor in get_neighbors(s, size):
            if known_grid[neighbor[0]][neighbor[1]] == 1:
                continue
            if neighbor in closed_set:
                continue

            # counter trick: init g if not seen in this search
            if search_val.get(neighbor, 0) < counter:
                g[neighbor] = float('inf')
                search_val[neighbor] = counter

            new_g = g[s] + 1
            if new_g < g[neighbor]:
                g[neighbor] = new_g
                tree[neighbor] = s
                h = h_of(neighbor)
                f = new_g + h
                pq.push(f, new_g, neighbor)

    # no path found
    if g[goal] == float('inf'):
        if return_closed:
            return None, expanded, closed_list
        return None, expanded

    # reconstruct path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = tree[current]
    path.append(start)
    path.reverse()

    if return_closed:
        return path, expanded, closed_list
    return path, expanded


def repeated_forward_astar(
    actual_maze: List[List[int]],
    start: Tuple[int, int] = START_NODE,
    goal: Tuple[int, int] = END_NODE,
    tie_breaking: str = "max_g",
    visualize_callbacks: Optional[Dict[str, Callable[[Tuple[int, int]], None]]] = None,
) -> Tuple[bool, List[Tuple[int, int]], int, int]:
    """
    Repeated Forward A*: plan with A*, move along path,
    replan when blocked cells discovered.
    Returns (found, executed_path, total_expanded, num_replans)
    """
    size = ROWS

    # agent's knowledge: -1=unknown, 0=known free, 1=known blocked
    known_grid = [[-1] * size for _ in range(size)]
    known_grid[start[0]][start[1]] = 0
    known_grid[goal[0]][goal[1]] = 0

    counter = 0
    g = {}
    search_val = {}

    current = start
    total_expanded = 0
    trajectory = [current]
    num_searches = 0

    while current != goal:
        # observe neighbors
        for n in get_neighbors(current, size):
            known_grid[n[0]][n[1]] = actual_maze[n[0]][n[1]]

        # run A*
        counter += 1
        num_searches += 1
        path, expanded = a_star_search(
            known_grid, current, goal, size, tie_breaking, counter, g, search_val
        )
        total_expanded += expanded

        if path is None:
            return False, trajectory, total_expanded, num_searches

        # follow the path until blocked or goal
        for i in range(1, len(path)):
            next_cell = path[i]
            current = next_cell
            trajectory.append(current)

            if current == goal:
                return True, trajectory, total_expanded, num_searches

            # observe from new position
            for n in get_neighbors(current, size):
                known_grid[n[0]][n[1]] = actual_maze[n[0]][n[1]]

            # check if remaining path is blocked
            path_blocked = False
            for j in range(i + 1, len(path)):
                if known_grid[path[j][0]][path[j][1]] == 1:
                    path_blocked = True
                    break

            if path_blocked:
                break

    return True, trajectory, total_expanded, num_searches


def show_astar_search(win: pygame.Surface, actual_maze: List[List[int]], algo: str, fps: int = 240, step_delay_ms: int = 0, save_path: Optional[str] = None) -> None:
    # [BONUS] TODO: Place your visualization code here.
    if save_path is None:
        save_path = f"vis_{algo}.png"

    pygame.image.save(win, save_path)
    print(f"Saved the visualization -> {save_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Q2: Repeated Forward A*")
    parser.add_argument("--maze_file", type=str, required=True,
                        help="Path to input JSON file containing a list of mazes")
    parser.add_argument("--output", type=str, default="results_q2.json",
                        help="Path to output JSON results file")
    parser.add_argument("--tie_braking", type=str, choices=["max_g", "min_g", "both"], default="both",
                        help="Tie-breaking variant to run (default: both)")
    parser.add_argument("--show_vis", action="store_true",
                        help="[Bonus] If set, show Pygame visualization for the selected maze")
    parser.add_argument("--maze_vis_id", type=int, default=0,
                        help="[Bonus] maze_id (index) 0 ... 49 among 50 grid worlds")
    parser.add_argument("--save_vis_path", type=str, default="q2-vis-max-g.png",
                        help="[Bonus] If set, save visualization to this PNG file")
    args = parser.parse_args()

    mazes = readMazes(args.maze_file)
    results: List[Dict] = []

    for maze_id in tqdm(range(len(mazes)), desc="Processing mazes"):
        entry: Dict = {"maze_id": maze_id}

        if args.tie_braking in ("max_g", "both"):
            t0 = time.perf_counter()
            found, executed, expanded, replans = repeated_forward_astar(
                actual_maze=mazes[maze_id],
                start=START_NODE,
                goal=END_NODE,
                tie_breaking="max_g"
            )
            t1 = time.perf_counter()

            entry["max_g"] = {
                "found": found,
                "path_length": len(executed) - 1 if found else -1,
                "expanded": expanded,
                "replans": replans,
                "runtime_ms": (t1 - t0) * 1000,
            }

        if args.tie_braking in ("min_g", "both"):
            t0 = time.perf_counter()
            found, executed, expanded, replans = repeated_forward_astar(
                actual_maze=mazes[maze_id],
                start=START_NODE,
                goal=END_NODE,
                tie_breaking="min_g"
            )
            t1 = time.perf_counter()

            entry["min_g"] = {
                "found": found,
                "path_length": len(executed) - 1 if found else -1,
                "expanded": expanded,
                "replans": replans,
                "runtime_ms": (t1 - t0) * 1000,
            }

        results.append(entry)

    if args.show_vis:
        pygame.init()
        win = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("Repeated Forward A* Visualization")
        clock = pygame.time.Clock()
        selected_maze = mazes[args.maze_vis_id]
        current_algo = "max_g"
        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
        running = True
        while running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_1:
                        current_algo = "max_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
                    elif event.key == pygame.K_2:
                        current_algo = "min_g"
                        show_astar_search(win, selected_maze, algo=current_algo, fps=240, step_delay_ms=0, save_path=args.save_vis_path)
            pygame.display.flip()

        pygame.quit()

    with open(args.output, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"Results for {len(results)} mazes written to {args.output}")


if __name__ == "__main__":
    main()
