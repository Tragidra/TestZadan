import json
import time
import numpy as np
from scipy import special
from scipy.ndimage import median

n = 10
class Solvers():
    def __init__(self):
        super(Solvers, self).__init__()
        a = np.load("s.npy")
        b = open('options.json')
        op = json.load(b)
        res1 = self.solver_a(a, op['shift'])
        res2 = self.solver_b(a, op['L'], op['W'])
        res3 = self.solver_c(res1,res2)
        print(median(res1))
        print(median(res2))
        print(median(res3))

    def solver_a(self, s, shift):
        b1 = np.power(s,0.5) #заменил возведение в степеньи
        in1 = np.zeros(np.shape(b1))
        calc_value = 1.5707963 * b1 * (special.k0(b1) * special.modstruve(1, b1) + special.k1(b1) * special.modstruve(0, b1)) + b1 * special.k0(b1) + shift
        in2 = np.where(b1 > 0.0005, in1, calc_value)
        in2 = np.where(b1 <= 0.0005, in2, np.pi / 2)
        return np.nan_to_num(in2 / b1)

    def solver_b(self, s, l, w):
        r_array: np.ndarray = self.get2DArray(l, w, n)

        s_sqrt = np.sqrt(s)

        res = np.zeros([len(s_sqrt), len(s_sqrt[0])])

        for i in range(len(r_array)):
            for j in range(len(r_array[0])):
                mul_array = r_array * special.modstruve(0, (1/r_array.size)**2)
                el = mul_array[i, j]

                calc = special.modstruve(1, el)

                for s_i in range(len(s_sqrt)):
                    for s_j in range(len(s_sqrt[0])):
                        res[s_i, s_j] += s_sqrt[s_i, s_j] * calc

        return self.solver_a(res, 0)

    def solver_c(self, res1,res2):
        return np.nan_to_num(res1 / res2 + res1)

    def get2DArray(self, L, W, n) -> np.ndarray:
        shape = (2 * n + 1, 2 * n + 1)
        return np.fromfunction(lambda i, j: self.r_factory(i, j, W, L, n), shape)

    def r_factory(self, i: int, j: int, W: float, L: float, n: int):
        gx = 2 * ((i - n) * W)
        gy = 2 * ((j - n) * L)
        return np.sqrt(gx * gx + gy * gy)


#ts = time.perf_counter() #замеры для подсчёта потраченного времени
Solvers()
#tf = time.perf_counter()
#print(tf-ts)
