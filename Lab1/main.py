import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import csv

file = r"./14/r1z1.csv"

data = []
with open(file) as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == 'X':
            continue
        data.append(*row)

data = sorted(map(float, data))
npArray = np.array(data)

# data
size = len(data)
print("Объём данных ", size)

# max and min
min_num = data[0]
max_num = data[size - 1]
print("min:", min_num, "\nmax:", max_num)

# scope
scope = max_num - min_num
range_data = np.ptp(npArray)
print("Размах:", scope, "Функцией:", range_data)

# average
avg = sum(data) / size
mean_data = np.mean(npArray)
print("Среднее значение:", avg, "Функцией:", mean_data)

# sample variance
variance_data = stats.describe(data).variance

sample_variance = 0
for x in data:
    sample_variance += ((x - avg) ** 2) / (size - 1)

print("Дисперсия выборочной:", sample_variance,
      "Функцией:", variance_data)

# Population variance
population_variancef = np.var(data)
population_variance = 0
for x in data:
    population_variance += (x - avg) ** 2 / size
print("Дисперсия популяции:", population_variance,
      "Функцией:", population_variancef)

# standard deviation
standard_deviation = population_variance ** 0.5
standard_deviationf = np.std(npArray)

print("Стандартного отклонения:", standard_deviation,
      "Функцией:", standard_deviationf)

# medium
medium = data[size // 2] if size % 2 == 0 else (data[size // 2] + data[size // 2 + 1]) / 2
median_data = np.median(npArray)
print("Медиана:", medium,
      "Функцией:", median_data)

# mode
mode = max(set(data), key=data.count)
mode_data = stats.mode(data).mode
print("Мода:", mode,
      "Функцией:", mode_data)

# symmetry coefficients
skewness = sum(((x - avg) / standard_deviation) ** 3 for x in data) * size / ((size - 1) * (size - 2))
skewness_data = stats.skew(data)
print("Коэффициент симметрии:", skewness,
      "Функцией:", skewness_data)

# quantiles (25%, 50%, 75%)

first_quantile = data[size // 4] if size % 4 == 0 else (data[size // 4] + data[size // 4 + 1]) / 2
medium_quantile = medium
last_quantile = data[3 * size // 4] if size % 4 != 0 else (data[3 * size // 4] + data[3 * size // 4 + 1]) / 2

quantile_25 = np.percentile(npArray, 25)
quantile_50 = np.percentile(npArray, 50)
quantile_75 = np.percentile(npArray, 75)
print("25% 50% 75%", first_quantile, medium_quantile, last_quantile,
      "\n25% 50% 75% f", quantile_25, quantile_50, quantile_75)
# interquartile latitude
interquartile = last_quantile - first_quantile
iqr_data = np.percentile(data, 75) - np.percentile(data, 25)
iqr_dataf = np.percentile(data, 75) - np.percentile(data, 25)
print("Интерквартильная ширина: ", iqr_data,
      "Функцией:", iqr_dataf)

# graphics
# Построение частотной гистограммы
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.hist(data, bins=30, color='blue', edgecolor='black')
plt.title('Частотная гистограмма')

# Построение вероятностной гистограммы с функцией плотности
plt.subplot(2, 2, 2)
sns.histplot(data, kde=True, color='blue')
plt.title('Вероятностная гистограмма с функцией плотности')

# Полигон распределения
plt.subplot(2, 2, 3)
sns.kdeplot(data, cumulative=True, color='red')
plt.title('Эмпирическая функция (функция распределения)')

# Нанесение нормального распределения на эмпирическую функцию
x = data
y = norm.cdf(x, loc=medium, scale=standard_deviation)
plt.plot(x, y, color='blue')
plt.title('Нормальное распределение на эмпирической функции')

# Построение box plot
plt.subplot(2, 2, 4)
plt.boxplot(data, vert=True, patch_artist=True, showmeans=True, showfliers=True)
plt.title('Box Plot')
plt.xlabel('Значения')
plt.show()
