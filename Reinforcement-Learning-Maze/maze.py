from PyQt5.QtCore import QPoint, QRectF, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QPainter, QFont, QPainterPath
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget, QMainWindow, QDesktopWidget
import numpy as np

from model import Direct, QLearningModel, SarsaModel
from envs import Env

class Grid(QWidget):
    """ A square grid in the maze """
    def __init__(self, side, isEnd=False, isBlock=False):
        super().__init__()
        self.hasDot = False  # whether agent is in this grid
        self.score = None
        self.isEnd = isEnd
        self.isBlock = isBlock
        self.side = side  # length of side
        self.action = None
        
        self.setFixedSize(self.side, self.side)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self._center = QPoint(self.side // 2, self.side // 2)

    def setHasDot(self, hasDot: bool):
        self.hasDot = hasDot
        self.update()
    
    def setScore(self, score, action=None):
        # 'action' is the direct which has the biggest score
        self.score = score
        if score is not None:
            self.score = float(format(score, ".2f"))
        self.action = action
        self.update()
    
    def _setColor(self, color: str):
        self.setStyleSheet("background-color: {}".format(color))
    
    def paintEvent(self, event):
        if self.isBlock:
            self._setColor("#aaaaaa")
            return
        painter = QPainter(self)
        if self.isEnd:
            self._setColor("rgb(128,128,128)")
            painter.setPen(QColor("#ffffff"))
            painter.drawRect(int(self.side * 0.1), int(self.side * 0.1), int(self.side * 0.8), int(self.side * 0.8))
        
        # Draw dot
        if self.hasDot:
            painter.setBrush(QColor("#0000ff"))
            painter.setPen(QColor("#0000ff"))
            painter.drawEllipse(self._center, self.side // 6, self.side // 6)
        
        # Draw score
        if self.score is not None:
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont('Arial', self.side // 8))
            painter.drawText(event.rect(), Qt.AlignCenter, str(self.score))
            if self.score >= 0:
                score = self.score if self.score <= 1 else 1
                self._setColor("rgba(0,255,0,{})".format(score))
            elif self.score < 0:
                score = self.score if self.score >= -1 else -1
                self._setColor("rgba(255,0,0,{})".format(-score))
        
        # Draw direction arrow
        actionSide = self.side * 0.2
        if self.action == Direct.LEFT:
            rect = QRectF(0, self.side * 0.4, actionSide, actionSide)
            painterPath = QPainterPath()
            painterPath.moveTo(rect.left(), rect.top() + rect.height() / 2)
            painterPath.lineTo(rect.bottomRight())
            painterPath.lineTo(rect.topRight())
            painterPath.lineTo(rect.left(), rect.top() + rect.height() / 2)
            painter.fillPath(painterPath, QBrush(QColor ("#ffffff")))
        elif self.action == Direct.RIGHT:
            rect = QRectF(self.side - actionSide, self.side * 0.4, actionSide, actionSide)
            painterPath = QPainterPath()
            painterPath.moveTo(rect.right(), rect.top() + rect.height() / 2)
            painterPath.lineTo(rect.topLeft())
            painterPath.lineTo(rect.bottomLeft())
            painterPath.lineTo(rect.right(), rect.top() + rect.height() / 2)
            painter.fillPath(painterPath, QBrush(QColor ("#ffffff")))
        elif self.action == Direct.UP:
            rect = QRectF(self.side * 0.4, 0, actionSide, actionSide)
            painterPath = QPainterPath()
            painterPath.moveTo(rect.left() + (rect.width() / 2), rect.top())
            painterPath.lineTo(rect.bottomLeft())
            painterPath.lineTo(rect.bottomRight())
            painterPath.lineTo(rect.left() + (rect.width() / 2), rect.top())
            painter.fillPath(painterPath, QBrush(QColor ("#ffffff")))
        elif self.action == Direct.DOWN:
            rect = QRectF(self.side * 0.4, self.side - actionSide, actionSide, actionSide)
            painterPath = QPainterPath()
            painterPath.moveTo(rect.left() + rect.width() / 2, rect.bottom())
            painterPath.lineTo(rect.topLeft())
            painterPath.lineTo(rect.topRight())
            painterPath.lineTo(rect.left() + rect.width() / 2, rect.bottom())
            painter.fillPath(painterPath, QBrush(QColor ("#ffffff")))
            

class Maze(QMainWindow):
    def __init__(self, env: Env, Model):
        super().__init__()
        self.m = env.m
        self.n = env.n
        self.startPos = env.startPos
        self.endPosList = env.endPosList
        self.endScoreList = env.endScoreList
        self.blockPosList = env.blockPosList
        self.punish = env.punish
        self.fresh_time = env.fresh_time

        self.box = QGridLayout()  # the layout that grids stay in
        self.gridList = []  # save all grids
        self.startGrid = None

        self.initBox()
        self.initUI()

        self.currentPos = env.startPos
        self.validDirect = self.getValidDirect()

        self.timer = QTimer()
        self.timer.timeout.connect(self.train)
        self.timer.start(1000)  # Start training after 1 second
        self.model = Model(env)
    
    def initBox(self):
        # build all square grids
        self.box.setSpacing(5)
        side = int(min(self.height() / (self.m + 1), self.width() / (self.n + 1)))
        index = 0
        for i in range(self.m):
            for j in range(self.n):
                isEnd = (i, j) in self.endPosList
                isBlock = (i, j) in self.blockPosList
                grid = Grid(side, isEnd, isBlock)

                if (i, j) == self.startPos:
                    self.startGrid = grid
                    grid.setHasDot(True)
                elif isEnd:
                    grid.setScore(self.endScoreList[self.endPosList.index((i, j))])
                
                self.box.addWidget(grid, i, j)
                self.gridList.append(grid)
                index += 1

    def initUI(self):
        self.resize(500, 500)
        self.setWindowTitle('强化学习 - 迷宫')
        self.setStyleSheet("background: black")

        mainHBox = QHBoxLayout()
        mainVBox = QVBoxLayout()
        mainHBox.addStretch(1)
        mainHBox.addLayout(mainVBox)
        mainHBox.addStretch(1)
        mainVBox.addStretch(1)
        mainVBox.addLayout(self.box)
        mainVBox.addStretch(1)

        widget = QWidget()
        self.setCentralWidget(widget)
        widget.setLayout(mainHBox)

        # move this window to the center of the screen
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def getValidDirect(self):
        # All directions that can be move forward
        validDirect = []
        if self.currentPos[0] - 1 >= 0 and (self.currentPos[0] - 1, self.currentPos[1]) not in self.blockPosList:
            validDirect.append(Direct.UP)
        if self.currentPos[0] + 1 < self.m and (self.currentPos[0] + 1, self.currentPos[1]) not in self.blockPosList:
            validDirect.append(Direct.DOWN)
        if self.currentPos[1] - 1 >= 0 and (self.currentPos[0], self.currentPos[1] - 1) not in self.blockPosList:
            validDirect.append(Direct.LEFT)
        if self.currentPos[1] + 1 < self.n and (self.currentPos[0], self.currentPos[1] + 1) not in self.blockPosList:
            validDirect.append(Direct.RIGHT)
        return validDirect
    
    @staticmethod
    def randomDirect(direct):
        # 80% -> forward, 10% -> left, 10% -> right, 0% -> backward
        if direct == Direct.UP:
            direct = np.random.choice([Direct.UP, Direct.DOWN, Direct.LEFT, Direct.RIGHT], p=[0.8, 0.0, 0.1, 0.1])
        elif direct == Direct.DOWN:
            direct = np.random.choice([Direct.UP, Direct.DOWN, Direct.LEFT, Direct.RIGHT], p=[0.0, 0.8, 0.1, 0.1])
        elif direct == Direct.LEFT:
            direct = np.random.choice([Direct.UP, Direct.DOWN, Direct.LEFT, Direct.RIGHT], p=[0.1, 0.1, 0.8, 0.0])
        elif direct == Direct.RIGHT:
            direct = np.random.choice([Direct.UP, Direct.DOWN, Direct.LEFT, Direct.RIGHT], p=[0.1, 0.1, 0.0, 0.8])
        return direct
    
    def moveTo(self, direct):
        # Move in maze
        direct = self.randomDirect(direct)  # Get actual direct
        
        current_index = self.currentPos[0] * self.n + self.currentPos[1]
        if direct in self.validDirect:
            currentGrid = self.gridList[self.currentPos[0] * self.n + self.currentPos[1]]
            currentGrid.setHasDot(False)

            # Compute position of next grid
            if direct == Direct.LEFT:
                self.currentPos = (self.currentPos[0], self.currentPos[1] - 1)
            elif direct == Direct.RIGHT:
                self.currentPos = (self.currentPos[0], self.currentPos[1] + 1)
            elif direct == Direct.UP:
                self.currentPos = (self.currentPos[0] - 1, self.currentPos[1])
            elif direct == Direct.DOWN:
                self.currentPos = (self.currentPos[0] + 1, self.currentPos[1])

            current_index = self.currentPos[0] * self.n + self.currentPos[1]
            self.gridList[current_index].setHasDot(True)
            self.validDirect = self.getValidDirect()

            if self.isDone():
                return current_index, self.gridList[current_index].score, True
        
        # forward is a block grid
        return current_index, self.punish, False
    
    def isDone(self):
        # Whether the dot is in a end grid
        currentGrid = self.gridList[self.currentPos[0] * self.n + self.currentPos[1]]
        return currentGrid.isEnd
    
    def reset(self):
        # Restart a game using the same env
        currentGrid = self.gridList[self.currentPos[0] * self.n + self.currentPos[1]]
        currentGrid.setHasDot(False)
        self.startGrid.setHasDot(True)
        self.currentPos = self.startPos
        self.validDirect = self.getValidDirect()
        self.model.current_state = self.startPos[0] * self.n + self.startPos[1]
    
    def train(self):
        # Train given model
        if isinstance(self.model, SarsaModel):
            # Policy of Sarsa Model
            if self.model.next_action is None:
                action = self.model.chooseAction(self.model.current_state)
            else:
                action = self.model.next_action
            
            next_state, reward, done = self.moveTo(action)
            self.model.next_action = self.model.chooseAction(next_state)
            self.model.learn(self.model.current_state, action, reward, next_state, self.model.next_action, done)
        
        elif isinstance(self.model, QLearningModel):
            # Policy of Q-Learning Model
            action = self.model.chooseAction(self.model.current_state)
            next_state, reward, done = self.moveTo(action)
            self.model.learn(self.model.current_state, action, reward, next_state, done)

        else:
            return
        
        # Update scores of grids
        for i, grid in enumerate(self.gridList):
            if grid.isBlock or grid.isEnd:
                continue
            grid.setScore(self.model.q_table[i].max(), self.model.q_table[i].argmax())

        if done:
            self.reset()

        if self.model.current_epoch < self.model.n_epoch:
            self.timer.start(self.fresh_time)
        else:
            print("Finish!")
            self.timer.stop()
    
    # For KeyBoard input test
    def keyPressEvent(self, event):
        # if event.key() == Qt.Key_Up:
        #     self.moveTo(Direct.UP)
        # elif event.key() == Qt.Key_Down:
        #     self.moveTo(Direct.DOWN)
        # elif event.key() == Qt.Key_Left:
        #     self.moveTo(Direct.LEFT)
        # elif event.key() == Qt.Key_Right:
        #     self.moveTo(Direct.RIGHT)
        pass
