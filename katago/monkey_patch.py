import sys
import os
import numpy as np
import torch

###################################################
# import katago modules

# Temporarily enable absolute imports for original_katago/*.py.
kata_path = os.path.join(os.path.dirname(__file__), 'original_katago')
original_sys_path = sys.path.copy()
sys.path = [kata_path] + sys.path

from gamestate import GameState as KataGameState
from board import Board as KataBoard
from load_model import load_model as kata_load_model

sys.path = original_sys_path

# Both katago and tamago define a module named 'board'.
# To ensure tamago's 'board' is imported next,
# remove katago's 'board' from the cached modules.
del sys.modules['board']

###################################################
# set BOARD_SIZE

from board.set_constant import respect_size_option
respect_size_option(sys.argv)

###################################################
# patch copy_board

from board.go_board import copy_board

def patched_copy_board(dst, src):
    copy_board(dst, src)
    copy_kata_gamestate(dst.kata_gamestate, src.kata_gamestate)

def copy_kata_gamestate(dst, src):
    dst.board_size = src.board_size
    dst.board =      src.board.copy()
    dst.moves =      src.moves.copy()
    dst.boards =     src.boards.copy()
    # dst.boards =     [board.copy() for board in src.boards]
    dst.rules =      src.rules.copy()
    dst.redo_stack = src.redo_stack.copy()

def copied_kata_gamestate(src):
    dst = KataGameState(src.board_size, src.rules)
    copy_kata_gamestate(dst, src)
    return dst

sys.modules['board.go_board'].copy_board = patched_copy_board

###################################################
# patch GoBoard

from board.go_board import GoBoard
from board.stone import Stone
from board.coordinate import Coordinate
from common.print_console import print_err

class PatchedGoBoard(GoBoard):
    def clear(self):
        super().clear()
        self._clear_kata_gamestate()

    def put_stone(self, pos, color):
        super().put_stone(pos, color)
        self._kata_put_stone(pos, color)

    def put_handicap_stone(self, pos, color):
        super().put_handicap_stone(pos, color)
        self._kata_put_stone(pos, color)

    def display(self, sym=0):
        super().display(sym)
        print_err('(KataGo GameState)')
        print_err(self.kata_gamestate.board.to_string() + '\n')

    #################################
    # util.

    def _clear_kata_gamestate(self):
        self.kata_gamestate = KataGameState(
            board_size=self.board_size,
            rules=KataGameState.RULES_CHINESE | {'whiteKomi': self.komi},
        )

    def _kata_put_stone(self, pos, color):
        board = self.kata_gamestate.board
        pla = KataBoard.BLACK if color == Stone.BLACK else KataBoard.WHITE
        xy = self._pos_to_xy(pos)
        loc = board.loc(*xy)
        if not board.would_be_legal(pla, loc):
            loc = KataBoard.PASS_LOC
        self.kata_gamestate.play(pla, loc)

    def _pos_to_xy(self, pos):
        coordinate = Coordinate(self.get_board_size())
        sgf_format = coordinate.sgf_format
        sgf_x, sgf_y = coordinate.convert_to_sgf_format(pos)
        x = sgf_format.index(sgf_x)
        y = sgf_format.index(sgf_y)
        return (x, y)

sys.modules['board.go_board'].GoBoard = PatchedGoBoard

###################################################
# patch generate_input_planes

from nn.feature import generate_input_planes

# enable attribute addition
class PatchedArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

def patched_generate_input_planes(board, color, sym=0):
    input_data = PatchedArray(generate_input_planes(board, color, sym))
    input_data.kata_gamestate = copied_kata_gamestate(board.kata_gamestate)
    return input_data

sys.modules['nn.feature'].generate_input_planes = patched_generate_input_planes

###################################################
# patch MCTSTree

from mcts.tree import MCTSTree
from nn.utility import get_torch_device

class PatchedMCTSTree(MCTSTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kata_model = None
    def load_katago_model(self, model_file_path, size, use_gpu):
        device = get_torch_device(use_gpu=use_gpu)
        model, _, _ = kata_load_model(model_file_path, False, device=device, pos_len=size, verbose=True)
        self.kata_model = model
    def mini_batch_inference(self, use_logit=False):
        # fixme: inefficient.
        # It's better to write a batch version of kata_get_policy_value.
        gss = [p.kata_gamestate for p in self.batch_queue.input_plane]
        outs = [kata_get_policy_value(gs, self.kata_model) for gs in gss]
        policies, values = zip(*outs)
        # np.array to suppress warning
        raw_policy = torch.tensor(np.array(policies))
        value_data = torch.tensor(np.array(values))
        if use_logit:
            raw_policy = torch.log(raw_policy)
        return raw_policy, value_data

def kata_get_policy_value(gs, model):
    outputs = gs.get_model_outputs(model)
    policy = outputs['policy0']
    win, lose, draw = outputs['value']
    value = [lose, draw, win]  # for tamago compatibility
    return policy, value

sys.modules['mcts.tree'].MCTSTree = PatchedMCTSTree

###################################################
# ignore tamago model

from nn.utility import load_network

def patched_load_network(*args, **kwargs):
    return None

sys.modules['nn.utility'].load_network = patched_load_network
