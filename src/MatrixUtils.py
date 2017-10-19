import numpy as np
from ScLogger import ScLogger


class MatrixUtils:
    # from https://stackoverflow.com/a/30418912
    @staticmethod
    def findSubMatrix(bigMatrix, buildingSize):
        minSize = buildingSize
        # tt = np.array([[1, 0, 0, 0, 0, 0, 6, 7, 8, 9], [0, 1, 0, 0, 0, 0, 6, 7, 8, 9], [0, 0, 1, 0, 0, 0, 6, 7, 8, 9],
        #                [0, 0, 0, 1, 0, 0, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 6, 7, 8, 9],
        #                [0, 0, 0, 0, 0, 0, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 6, 7, 8, 9],
        #                [0, 0, 0, 0, 0, 0, 6, 7, 8, 9]])
        nrows = 10
        ncols = 10
        area_max = (0, [])
        w = np.zeros(dtype=int, shape=bigMatrix.shape)
        h = np.zeros(dtype=int, shape=bigMatrix.shape)

        for r in range(nrows):
            for c in range(ncols):
                if bigMatrix[r][c] != 0:
                    continue
                if r == 0:
                    h[r][c] = 1
                else:
                    h[r][c] = h[r - 1][c] + 1
                if c == 0:
                    w[r][c] = 1
                else:
                    w[r][c] = w[r][c - 1] + 1
                minw = w[r][c]
                minh = h[r][c]
                for dh in range(h[r][c]):
                    minw = min(minw, w[r - dh][c])
                    minh = min(minh, h[r - dh][c])
                    area = (dh + 1) * minw
                    # print("minh " + str(int(minh)))
                    # print("minw " + str(int(minw)))
                    if area > area_max[0]:# and minSize < minw and minSize < minh:
                        area_max = (area, [(r - dh, c - minw + 1, r, c)])
                        break
        np.set_printoptions(threshold='nan')
        ScLogger.log(bigMatrix)
        ScLogger.log("###############################")
        for t in area_max[1]:
            ScLogger.log('Cell 1:(row {}, col {}) and Cell 2:( row {}, col {})'.format(*t))
            return t[0], t[1]

# [[1 0 0 0 0 0 6 7 8 9],
#  [0 1 0 0 0 0 6 7 8 9],
#  [0 0 1 0 0 0 6 7 8 9],
#  [0 0 0 1 0 0 6 7 8 9],
#  [0 0 0 0 0 0 6 7 8 9],
#  [0 0 0 0 0 0 6 7 8 9],
#  [0 0 0 0 0 0 6 7 8 9],
#  [0 0 0 0 0 0 6 7 8 9],
#  [0 0 0 0 0 0 6 7 8 9],
#  [0 0 0 0 0 0 6 7 8 9]]
