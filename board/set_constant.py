"""board.constantの定数を実行時に設定するハック
"""
import sys

from board.constant import BOARD_SIZE as DUMMY_BOARD_SIZE

# respect_size_option(sys.argv) で
# --size オプションの値を BOARD_SIZE に設定
def respect_size_option(argv):
    key = '--size'
    if key in argv:
        i = argv.index(key)
        size = int(argv[i + 1])
        set_size(size)

def set_size(size):
    BC = sys.modules['board.constant']
    # 碁盤のサイズ
    BC.BOARD_SIZE = size
    # 盤外のサイズ
    BC.OB_SIZE = 1
    # 連の最大数
    BC.STRING_MAX = int(0.8 * BC.BOARD_SIZE * (BC.BOARD_SIZE - 1) + 5)
    # 隣接する連の最大数
    BC.NEIGHBOR_MAX = BC.STRING_MAX
    # 連を構成する石の最大数
    BC.STRING_POS_MAX = (BC.BOARD_SIZE + BC.OB_SIZE * 2) ** 2
    # 呼吸点の最大数
    BC.STRING_LIB_MAX = (BC.BOARD_SIZE + BC.OB_SIZE * 2) ** 2
    # 連を構成する石の座標の番兵
    BC.STRING_END = BC.STRING_POS_MAX - 1
    # 呼吸点の番兵
    BC.LIBERTY_END = BC.STRING_LIB_MAX - 1
    # 隣接する敵連の番兵
    BC.NEIGHBOR_END = BC.NEIGHBOR_MAX - 1

    # 着手履歴の最大数
    BC.MAX_RECORDS = (BC.BOARD_SIZE ** 2) * 3
