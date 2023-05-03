from enum import IntEnum
from typing import Tuple, List

import numpy as np

TOKEN_CHARS = [
        ' ',
        'r',
        'b',
        'g',
        'y',
        '#',
        'R',
        'B',
        'G',
        'Y'
    ]
class Token(IntEnum):
    empty = 0
    red = start_tokens = 1
    blue = 2
    green = 3
    yellow = end_tokens = 4
    highlighted = 5
    highlighted_red = 6
    highlighted_blue = 7
    highlighted_green = 8
    highlighted_yellow = 9

    debug = 10

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

class BoardState:
    IN_PLAY = 0
    IN_PLAY_WITH_CONNECTIONS = 1
    VICTORY = 2
    TIE = 3

    def __init__(self):
        self.type = BoardState.IN_PLAY
        self.terminal = False
        self.state_display = None
        self.victor = None

class Board:
    def __init__(self, size : Tuple[int, int, int], token_history_len : int = 4, connect_n : int = 4, debug=False):
        self.tokens : np.ndarray = np.array([[[Token.empty,]*size[0]]*size[1]]*size[2], dtype='uint8')
        self.action_history = []
        self.token_history_len = token_history_len
        self.token_history_i = 1
        if self.token_history_len is None:
            self.token_history = np.array([self.tokens] * connect_n)
        else:
            self.token_history = np.array([self.tokens] * token_history_len)
        self.size : Tuple[int, int, int] = size
        self.gravity : Gravity = Gravity.cli_default
        self.connect_n = connect_n
        self.debug = debug
        self.state = BoardState()

    def __repr__(self):
        return f"Board({self.size})\n  tokens=\n{self.tokens}\n  gravity=\n{self.gravity}"
    
    @staticmethod
    def tokens_cli_str(tokens):
        s = str(tokens)
        for i, c in enumerate(TOKEN_CHARS):
            s = s.replace(str(i), c)
        return s

    def __str__(self):
        return Board.tokens_cli_str(self.tokens) + f"\nGravity is {self.gravity.name} ({self.gravity.value})"
    
    def log(self, s : str):
        if self.debug:
            print(s)

    def highlight_row(self, x : int, y : int, gravity=None) -> np.ndarray:
        if gravity == None:
            gravity = self.gravity
        loc = Board.plane_loc(gravity, x, y)
        # inefficient yes but that doesn't matter
        display_tokens = self.tokens.copy()
        for i in range(len(display_tokens[loc])):
            if display_tokens[loc][i] < Token.highlighted:
                display_tokens[loc][i] += Token.highlighted
        return display_tokens


    def populate_randomly(self, max_token=Token.end_tokens):
        self.tokens : np.ndarray = np.random.randint(0, max_token+1, size=self.size, dtype='uint8')

    def update_token_history(self):
        if self.token_history_len is None:
            self.token_history.append(self.tokens)
        else:
            self.token_history[self.token_history_i % self.token_history_len] = self.tokens
        self.token_history_i += 1

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

        self.log('gravity applied')

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
            self.log(f'placing {placed_token} at {loc} in row {tokens}')
            if self.gravity >= 0:
                for i in range(n-1):
                    if tokens[-i] != Token.empty:
                        self.tokens[loc][i-1] = placed_token
                        self.log(f'placed {placed_token} at {loc}:{i-1} (g+)')
                        token_placed = True
                        break
                if not(token_placed):
                    self.tokens[loc][-1] = placed_token
                    self.log(f'placed {placed_token} at {loc}:TOP')
            else:
                for i in range(n, 1, -1):
                    if tokens[i] != Token.empty:
                        self.tokens[loc][i+1] = placed_token
                        self.log(f'placed {placed_token} at {loc}:{i+1} (g-)')
                        token_placed = True
                        break
                if not(token_placed):
                    self.tokens[loc][0] = placed_token
                    self.log(f'placed {placed_token} at {loc}:BOTTOM')

            self.apply_gravity() # update with new placed_token

            self.action_history.append((
                placed_token,
                loc,
                gravity
            ))
            self.update_token_history()
            return True
        else:
            self.log(f'unplaceable: {placed_token} at {loc}')
            return False

    def connect(self, n : int, dimensions = 0b11111) -> Tuple[List[int], List[List[int]], np.ndarray]:
        """returns a list of connect-n lines on the board

        n : the minimum length of a connection
        dimensions : an optional flag describing acceptable connections:
          0b0000001: 1D lines
          0b0000010: 2D diagonals
          0b0000100: 3D diagonals
          0b0010000: nD spacetime-diagonals (1 through 3)
          0b0011000: 4D spacetime-diagonals
          

        returns: a tuple of
            - a list of the # of connections per token, e.g. [0, n_connections for Token.red, n_connections for Token.blue]
            - a list of (a list of n_connections of each type) for each of Token.red, Token.blue
            - an np.ndarray of the board with highlighted cols
        """
        if dimensions != 0b11111:
            raise NotImplementedError


        N_CONNECTION_TYPES = 4
        valid_tokens = [Token(i) for i in range(Token.start_tokens,Token.end_tokens + 1)]
        connections = [[0] * N_CONNECTION_TYPES] * Token.end_tokens
        n_connections = [0] * Token.end_tokens
        token_display = self.tokens
        if dimensions & 0b10000:
            if self.token_history_len is None:
                arr = self.token_history[-n:]
            else:
                arr = np.roll(self.token_history, -self.token_history_i, axis=0)
                """
                [0, 0, 0, 0]: 1
                [0, 1, 0, 0]: 2
                ...
                [0, 1, 2, 3]: 4 == 0
                [4, 1, 2, 3]: 5 == 1
                """
                arr = arr[-n:]
        else:
            arr = self.tokens

        shape = arr.shape
        #assert(shape[0:3] == self.size)
        #assert(shape[4] == n)

        count = 0
        # Check for lines along each axis
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3] - n + 1):
                        if np.all(arr[i, j, k, l:l+n] == arr[i, j, k, l]):
                            count += 1
                            self.log('foundline')
        # Check for n-dimensional diagonals
        for i in range(arr.shape[0] - n + 1):
            for j in range(arr.shape[1] - n + 1):
                for k in range(arr.shape[2] - n + 1):
                    for l in range(arr.shape[3] - n + 1):
                        if np.all(arr[i:i+n, j:j+n, k:k+n, l:l+n].diagonal() == arr[i, j, k, l]):
                            count += 1
                            self.log('founddiagonal')

        """
        starts = np.array(np.meshgrid(*[np.arange(0, s-n+1) for s in shape])).T.reshape(-1, 4)
        offsets = np.array(np.meshgrid(*[np.arange(0, n) for _ in range(4)])).T.reshape(-1, 4)
        idxs = starts[:, None, :] + offsets
        connections = arr[idxs[:, :, 0], idxs[:, :, 1], idxs[:, :, 2], idxs[:, :, 3]]
        valid_connections = np.all(connections == connections[:, [0]], axis=1)
        count = np.sum(valid_connections)
        """

        #count = unique_idxs.shape[0]
        


        pass

    def check_victory_conditions(self):
        n_connections, connections, display_board = self.connect(self.connect_n)
        if any(n_connection > 0 for n_connection in n_connections):
            winners = np.argmax(n_connections)
            if len(winners) == 1:
                self.state.type = BoardState.VICTORY
                self.state.terminal = True
                self.state.victor = Token(winners[0])
        

