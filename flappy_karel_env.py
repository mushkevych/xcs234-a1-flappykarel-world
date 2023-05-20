from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional, Tuple, Dict

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

NORTH_EAST = 0  # right and up
SOUTH_EAST = 1  # right and down
SOUTH = 2       # falling down one square

MAP_NAME_7x5_S2 = "world_7x5_s2"
MAP_NAME_7x5_S3 = "world_7x5_s3"
MAP_NAME_7x7_S3 = "world_7x7_s3"

# Map tiles:
# R - red
# U - unshaded
# S - starting tile
# G - goal/green tile
MAPS = {
    MAP_NAME_7x5_S2: [
        "RRRRR",
        "SURUU",
        "UURUU",
        "UUUUG",
        "URUUU",
        "URUUU",
        "RRRRR",
    ],
    MAP_NAME_7x5_S3: [
        "RRRRR",
        "UURUU",
        "SURUU",
        "UUUUG",
        "URUUU",
        "URUUU",
        "RRRRR",
    ],
    MAP_NAME_7x7_S3: [
        "RRRRRRR",
        "UURUUUU",
        "SURUUUU",
        "UUUUUUG",
        "URUUUUU",
        "URUUUUU",
        "RRRRRRR",
    ],
}

REWARD_SCHEDULE_A: Dict[bytes, float] = {
    b"R": -5.0,
    b"G": +5.0,
    b"U": -4.0,
    b"S": -4.0,
}

REWARD_SCHEDULE_B: Dict[bytes, float] = {
    b"R": -5.0,
    b"G": +5.0,
    b"U": -1.0,
    b"S": -1.0,
}

REWARD_SCHEDULE_C: Dict[bytes, float] = {
    b"R": -5.0,
    b"G": +5.0,
    b"U": 0.0,
    b"S": 0.0,
}

REWARD_SCHEDULE_D: Dict[bytes, float] = {
    b"R": -5.0,
    b"G": +5.0,
    b"U": +1.0,
    b"S": +1.0,
}


# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(-1, 1), (1, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "R":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board = np_random.choice(["U", "R"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


class FlappyKarelEnv(Env):
    """
    Flappy Karel is a mobile game, where Karel the robot must dodge the red pillars of doom
    and flap its way to the green pasture.

    Flappy World
    * Each square on the Flappy World grid represents a single state which Karel may occupy

    * Squares shaded in red represent terminal states, taking an action from these squares will provide Karel with
    a reward of `rr` and `end the episode`

    *  Squares shaded in green represent the `goal state` and are also `terminal`, taking an action from these squares
    will provide Karel with a reward of `rg` and `end the episode`.

    *  Squares left unshaded represent `non-terminal states`, taking an action from these squares will provide Karel
    with a reward of `rs` Karel’s Movement


    ## Action Space
    The action shape is `(1,)` in the range `{0, 2}` indicating which direction to move the Karel.

    - NORTH_EAST = 0  # Karel can move right and up (e.g. starting from square 2 Karel can move to square 8)
    - SOUTH_EAST = 1  # Karel can move right and down (e.g. starting from square 2 Karel can move to square 10)
    - SOUTH = 2       # There are walls in each version of Flappy World represented by thicker edges.
                        If a move taken by Karel runs into a wall this will result in Karel falling down one square
                        (e.g. going in any direction from square 30 results in falling to square 31)

    *  Actions are deterministic and always succeed unless they cause Karel to run into a wall

    ## Observation Space
    The observation is a value representing Karel's current position as
    `current_row * nrows + current_col` (where both the row and col start at 0).

    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.

    The observation is returned as an `int()`.

    ## Starting State
    The episode starts with the player in state `[S]`

    ## Rewards
    - Discount factor γ = 0.9
    - rg = +5
    - rr = −5
    - rs = {−4, −1, 0, 1}

    ## There are three possible long-term behaviours:
    * terminate through a red square
    * terminate through a green square
    * do not ever terminate
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        map_as_text: List[str] = None,
        map_name: str = MAP_NAME_7x5_S3,
        reward_schedule: Dict[bytes, float] = None,
        are_borders_present: bool = True,
    ):
        if not reward_schedule:
            reward_schedule = REWARD_SCHEDULE_A
        self.reward_schedule = reward_schedule
        self.reward_range = (sorted(self.reward_schedule.values())[0], sorted(self.reward_schedule.values())[-1])

        self.render_mode = render_mode

        if map_as_text is None and map_name is None:
            map_as_text = generate_random_map()
        elif map_as_text is None:
            map_as_text = MAPS[map_name]
        self.world_map = np.asarray(map_as_text, dtype="c")
        self.nrow, self.ncol = self.world_map.shape

        self.nA = len([NORTH_EAST, SOUTH_EAST])
        self.nS = self.nrow * self.ncol

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.initial_state_distrib = np.array(self.world_map == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        def to_state(row: int, col: int) -> int:
            return row * self.ncol + col

        def inc(row: int, col: int, action: int) -> Tuple[int, int]:
            if are_borders_present and col + 1 > self.ncol - 1:
                # Karel has reached the boundary
                action = SOUTH
                self.lastaction = action

            if not are_borders_present and col + 1 > self.ncol - 1:
                # Karel has reached EAST end of the map, and needs to be teleported to the WEST of the map
                col = -1

            if action == NORTH_EAST:
                col = min(col + 1, self.ncol - 1)
                row = max(row - 1, 0)
            elif action == SOUTH_EAST:
                col = min(col + 1, self.ncol - 1)
                row = min(row + 1, self.nrow - 1)
            elif action == SOUTH:
                row = min(row + 1, self.nrow - 1)
            return row, col

        def update_probability_matrix(row: int, col: int, action: int) -> Tuple[int, float, bool]:
            newrow, newcol = inc(row, col, action)
            newstate = to_state(newrow, newcol)
            newstate_name = self.world_map[newrow, newcol]
            terminated = bytes(newstate_name) in b"GR"
            reward = self.reward_schedule[newstate_name]
            return newstate, reward, terminated

        for row in range(self.nrow):
            for col in range(self.ncol):
                state = to_state(row, col)
                for action in range(self.nA):
                    # transitions format:
                    # list[
                    #   tuple[
                    #     probability_of_transitioning_to_next_state: float,
                    #     next_state: int,
                    #     reward: float,
                    #     is_terminal: bool,
                    #   ]
                    # ]
                    transitions = self.P[state][action]

                    state_name = self.world_map[row, col]
                    if state_name in b"GR":
                        transitions.append((1.0, state, 0.0, True))
                    else:
                        transitions.append((1.0, *update_probability_matrix(row, col, action)))

        # pygame utils
        self.window_size = (min(64 * self.ncol, 512), min(64 * self.nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.red_state_img = None
        self.cracked_hole_img = None
        self.unshaded_state_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, float]]:
        """
        :return: Tuple[next_step:int, reward:float, is_terminal:bool, is_truncated:bool, auxiliary information dict]
        """
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, is_t = transitions[i]
        self.s = s
        self.lastaction = action

        if self.render_mode == "human":
            self.render()
        return int(s), r, is_t, False, {"prob": p}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[int, dict]:
        """
        :return: Tuple[Observation of the initial state,
                       dictionary contains auxiliary information complementing ``observation``]
        """
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Flappy Karel")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert self.window_surface is not None, "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.red_state_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.red_state_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.unshaded_state_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.unshaded_state_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in [
                    path.join(path.dirname(__file__), "img/elf_right.png"),
                    path.join(path.dirname(__file__), "img/elf_right.png"),
                    path.join(path.dirname(__file__), "img/elf_down.png"),
                ]
            ]

        desc = self.world_map.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.unshaded_state_img, pos)
                if desc[y][x] == b"R":
                    self.window_surface.blit(self.red_state_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the Karel
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 0
        actor_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"R":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        else:
            self.window_surface.blit(actor_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        map_as_text: List[bytes] = self.world_map.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        map_as_text: List[List[str]] = [[c.decode("utf-8") for c in line] for line in map_as_text]
        map_as_text[row][col] = utils.colorize(map_as_text[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['North-East', 'South-East'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in map_as_text) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
