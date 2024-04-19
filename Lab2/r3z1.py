import csv
import numpy as np


def reader_data(file) -> list:
    data = []
    with open(file) as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == 'X':
                continue
            data.append(*row)
    return data


data = np.array(list(map(float, reader_data('r3z1.csv'))))
n = len(data)

theta = n / (sum(data))

print(theta)
