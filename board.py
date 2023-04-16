from enum import IntEnum
from typing import Tuple

import numpy as np

TOKEN_CHARS = [
        ' ',
        'r',
        'b',
        '#',
        'R',
        'B'
    ]
class Token(IntEnum):
    empty = 0
    red = 1
    blue = 2

    highlighted = 3
    highlighted_red = 4
    highlighted_blue = 5

    debug = 99

class Gravity(IntEnum):
    # names used for CLI output

    right = default = 1
    left = -1
    down = 2
    up = -2
    below = cli_default = 3
    above = -3

    def loc_copysign(self, x):
        return (x if self >= 0 else -x-1)
    
    def copysign(self, x):
        return (x if self >= 0 else -x)
    
    def loc_top(self):
        return (0 if self >= 0 else -1)
    
    def loc_bottom(self):
        return (-1 if self >= 0 else 0)
    
    def loc_transpose(self, loc):
        transpose_amt = abs(self) - 1
        return loc[transpose_amt:] + loc[:transpose_amt]

class Board:
    def __init__(self, size : Tuple[int, int, int], token_history_len : int = 4):
        self.tokens : np.ndarray = np.array([[[Token.empty,]*size[0]]*size[1]]*size[2], dtype='uint8')
        self.action_history = []
        self.token_history_len = token_history_len
        self.token_history = [self.tokens] * token_history_len
        self.size : Tuple[int, int, int] = size
        self.gravity : Gravity = Gravity.cli_default

    def __repr__(self):
        return f"Board({self.size})\n  tokens=\n{self.tokens}\n  gravity=\n{self.gravity}"

    # cli_utils    
    
    @staticmethod
    def tokens_cli_str(tokens):
        s = str(tokens)
        for i, c in enumerate(TOKEN_CHARS):
            s = s.replace(str(i), c)
        return s

    def __str__(self):
        
        return Board.tokens_cli_str(self.tokens) + f"\nGravity is {self.gravity.name} ({self.gravity.value})"
    
    

    def highlight_row(self, x : int, y : int, gravity=None) -> np.ndarray:
        if gravity == None:
            gravity = self.gravity
        loc = Board.plane_loc(gravity, x, y)
        # inefficient yes but that doesn't matter
        display_tokens = self.tokens.copy()
        for i in range(len(display_tokens[loc])):
            display_tokens[loc][i] += Token.highlighted
        return display_tokens


    def populate_randomly(self):
        self.tokens : np.ndarray = np.random.randint(0, 4, size=self.size, dtype='uint8')

    

    def set_gravity(self, direction):
        self.gravity : Gravity = Gravity(direction)
        self.apply_gravity()

    def apply_gravity(self):
        transpose = lambda loc : self.gravity.loc_transpose(loc)
        gravity_parity = self.gravity.copysign(1)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                column_loc = Board.plane_loc(self.gravity, i, j)
                col = self.tokens[column_loc]
                idxes, = np.where(col == Token.empty)
                prev_idx = self.gravity.loc_top()
                for idx in idxes[::(1 if self.gravity >= 0 else -1)]:
                    if prev_idx == idx:
                        continue
                    from_loc = transpose((i,j,slice(prev_idx, idx, gravity_parity)))
                    to_loc = transpose((i,j,slice(prev_idx+gravity_parity, idx+gravity_parity, gravity_parity)))
                    self.tokens[to_loc] = self.tokens[from_loc]
                    self.tokens[transpose((i,j,prev_idx))] = Token.empty
                    prev_idx = prev_idx + gravity_parity
                #idx = self.gravity.loc_bottom()

                    
        print('gravity applied')

    @staticmethod
    def plane_loc(gravity : Gravity, x : int, y : int) -> tuple:
        return gravity.loc_transpose((x, y, slice(None)))

    def is_placeable(self, loc : tuple, proposed_gravity : Gravity = None):
        # loc : Tuple[int | slice, int | slice, int | slice] (see output of plane_loc)
        if proposed_gravity is None:
            proposed_gravity = self.gravity

        return any(token == Token.empty for token in self.tokens[loc]) # is there an empty in this row?

    def place(self, placed_token : Token, loc : tuple, gravity : Gravity = None):
        if len(loc) == 2:
            loc = Board.plane_loc((self.gravity if gravity is None else gravity), *loc)
        elif len(loc) == 3:
            pass
        else:
            raise ValueError('argument loc should be a Tuple[int, int] or a tuple with 2 ints and one slice(None)')

        if self.is_placeable(loc, gravity):
            
            if gravity is not None:
                self.set_gravity(gravity) # also applies gravity

            tokens = self.tokens[loc]
            n = len(tokens)
            token_placed = False
            print(f'placing {placed_token} at {loc} in row {tokens}')
            if self.gravity >= 0:
                for i in range(n-1):
                    if tokens[-i] != Token.empty:
                        self.tokens[loc][i-1] = placed_token
                        print(f'placed {placed_token} at {loc}:{i-1} (g+)')
                        token_placed = True
                        break
                if not(token_placed):
                    self.tokens[loc][-1] = placed_token
                    print(f'placed {placed_token} at {loc}:TOP')
            else:
                for i in range(n, 1, -1):
                    if tokens[i] != Token.empty:
                        self.tokens[loc][i+1] = placed_token
                        print(f'placed {placed_token} at {loc}:{i+1} (g-)')
                        token_placed = True
                        break
                if not(token_placed):
                    self.tokens[loc][0] = placed_token
                    print(f'placed {placed_token} at {loc}:BOTTOM')


            self.apply_gravity() # update with new placed_token

            self.action_history.append((
                placed_token,
                loc,
                gravity
            ))
            self.token_history.append(self.tokens)
            return True
        else:
            print(f'unplaceable: {placed_token} at {loc}')
            return False



