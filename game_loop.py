from board import *

try:
    raw_size = input('Enter board size (e.g. 4x4x4): ')
    raw_size = raw_size.strip()
    if 'x' in raw_size:
        size = tuple(int(x.strip().replace('x', '')) for x in raw_size.split('x') if x != '')
    elif ',' in raw_size:
        size = tuple(int(x.strip().replace(',', '')) for x in raw_size.split(',') if x != '')
    else:
        size = tuple(int(x.strip()) for x in raw_size.split(' ') if x != '')
    board = Board(size)
except Exception as e:
    print(e)
    print('Defaulting to 4x4x4')
    board = Board((4, 4, 4))

tokens = [
    Token.red,
    Token.blue
]

while True:
    for token in tokens:
        print(f"{token.name}'s turn:")
        print(board)

        turn_completed = False
        while not(turn_completed):
            gravity_set = False
            while not(gravity_set):
                #gravity = input(f'Enter gravitational direction (currently {board.gravity.value}): ')
                gravity = input(f'Enter gravitational direction: ')
                if gravity == '':
                    gravity = None
                    gravity_set = True
                else:
                    try:
                        gravity = int(gravity)
                        if -3 <= gravity <= 3:
                            if gravity == 0:
                                gravity = None
                            else:
                                gravity = Gravity(gravity)
                            gravity_set = True
                        else:
                            print(f'gravity must be between -3 and 3.')
                    except ValueError:
                        print(f'invalid value {gravity} for gravity, try an integer between -3 and 3.')
            
            while True:
                try:
                    x = input(f'Enter plane x: ')
                    x = int(x)
                    break
                except ValueError as e:
                    continue
            while True:
                try:
                    y = input(f'Enter plane y: ')
                    y = int(y)
                    break
                except ValueError as e:
                    continue
            try:
                print('Place token at:')
                print(Board.tokens_cli_str(board.highlight_row(x, y, gravity)))
                grav_str = ('' if gravity is None else f' with gravity {gravity.name}')
                yes_no = input(f'Place token here{grav_str}? (y/n) ')
                if yes_no == '' or yes_no[0] == 'y':
                    token_placed = board.place(token, (x, y), gravity)
                    if token_placed:
                        turn_completed = True
                    else:
                        print('That row is full.')
            #except IndexError as e:
            #    print(f'{e}')
            #    continue
        
            except Exception as e:
                print(e)
                if input('show full traceback? (y/n)').strip() == 'y':
                    raise e
