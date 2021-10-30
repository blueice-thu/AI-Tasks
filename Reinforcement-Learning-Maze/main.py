import sys
from PyQt5.QtWidgets import QApplication

from maze import Maze
from model import QLearningModel, SarsaModel
from envs import env1, env2

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Select env and model here
    env = Maze(env1, SarsaModel)

    env.show()
    sys.exit(app.exec_())