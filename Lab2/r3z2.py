import csv
import math

import numpy as np
from scipy.stats import t


def reader_data(file) -> list:
    data = []
    with open(file) as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'X':
                continue
            data.append(*row)
    return data


data = np.array(list(map(float, reader_data('r3z2.csv'))))
n = len(data)

Q = 0.99

#Двусторонняя
alpha = (1 - Q) / 2
avg = sum(data) / n
s = math.sqrt(sum([(x - avg) ** 2 for x in data]) / (n - 1))
stud = t.ppf(1 - alpha, n - 1)

p_upper = avg + stud * s / math.sqrt(n)
p_lower = avg - stud * s / math.sqrt(n)
