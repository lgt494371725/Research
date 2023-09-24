import matplotlib.pyplot as plt
import numpy as np

def plot_lines(mat_w, mat_h):
    left_border = up_border = -0.5
    right_border, down_border = mat_w - 0.5, mat_h - 0.5
    plt.hlines([i - 0.5 for i in range(mat_h)], left_border, right_border, color='black')
    plt.vlines([i - 0.5 for i in range(mat_w)], up_border, down_border, color='black')

def visualize(map):
    plt.matshow(-map, cmap=plt.cm.hot)
    mat_w, mat_h = len(map[0]), len(map)
    plot_lines(mat_w, mat_h)
    # plt.fill_between([-0.5, 0.5], [-0.5], [0.5], color='red', alpha=0.7)
    # plt.fill_between([2.5,3.5], [-0.5], [0.5], color='red', alpha=0.7)
    # plt.fill_between([-0.5, 0.5], [3.5], [4.5], color='red', alpha=0.7)
    # plt.fill_between([0.5, 1.5], [1.5], [2.5], color='green', alpha=0.7)
    # plt.fill_between([-0.5, 0.5], [0.5], [2.5], color='yellow', alpha=0.7)
    # plt.fill_between([1.5, 2.5], [-0.5], [0.5], color='yellow', alpha=0.7)
    # plt.fill_between([0.5, 3.5], [3.5], [4.5], color='yellow', alpha=0.7)
    # plt.fill_between([1.5, 3.5], [1.5], [2.5], color='gray', alpha=0.7)
    plt.show()


def read_map(path):
    number = 0
    print("map: ", path)
    matrix = []
    with open(path) as f:
        for row, line in enumerate(f.readlines()[4:]):  # 前3行无用
            line = line.strip('\n')
            temp = []
            for col, alpha in enumerate(line):
                if alpha == '.':
                    temp.append(0)
                else:
                    number += 1
                    temp.append(1)
            matrix.append(temp)
    print(f"obstacle number:{number}")
    return np.array(matrix)

map = read_map(r"C:\Users\18959\OneDrive - The University of Tokyo\research\master_research\maps\0_lak101d.map")
visualize(map)