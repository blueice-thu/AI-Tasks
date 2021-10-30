class Env:
    def __init__(self, m, n, startPos, endPosList, endScoreList, blockPosList, punish, fresh_time):
        # A maze of (m, n)
        self.m = m
        self.n = n


        """
            Position setting: (x, y)

                0  1  2 ... (n-1)
                --------------------> y
            0   |
            1   |
            2   |
            ... |
            m-1 |
                V
                x
        """

        self.startPos = startPos

        # 'endPosList' and 'endScoreList' must be in order
        self.endPosList = endPosList
        self.endScoreList = endScoreList

        self.blockPosList = blockPosList

        self.punish = punish  # A negative number
        self.fresh_time = fresh_time  # time interval between two actions, micro second

env1 = Env(
    m = 3,
    n = 4,
    startPos = (2, 0),
    endPosList = [ (0, 3), (1, 3) ],
    endScoreList = [ 1.00, -1.00 ],
    blockPosList = [(1, 1)],
    punish = -0.1,
    fresh_time = 50
)

env2 = Env(
    m = 5,
    n = 5,
    startPos = (0, 0),
    endPosList = [ (2, 3), (3, 1), (3, 3), (4, 1) ],
    endScoreList = [ -1.0, -1.0, 0.5, 1.0 ],
    blockPosList = [ (1, 0), (1, 1) ],
    punish = -0.1,
    fresh_time = 50
)
