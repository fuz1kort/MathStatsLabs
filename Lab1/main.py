import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import norm

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

# Данные
size = len(data)
print("Объём выборки ", size)

# Минимум/Максимум
min_num = data[0]
max_num = data[size - 1]
print("min:", min_num, "\nmax:", max_num)

# Размах
scope = max_num - min_num
range_data = np.ptp(npArray)
print("Размах:", scope, "Функцией:", range_data)

# Простое среднее
avg = sum(data) / size
mean_data = np.mean(npArray)
print("Среднее значение:", avg, "Функцией:", mean_data)

# Выборочная дисперсия
variance_data = stats.describe(data).variance
sample_variance = 0
for x in data:
    sample_variance += ((x - avg) ** 2) / (size - 1)
print("Выборочная дисперсия:", sample_variance,
      "Функцией:", variance_data)

# Дисперсия популяции
population_variancef = np.var(data)
population_variance = 0
for x in data:
    population_variance += (x - avg) ** 2 / size
print("Дисперсия популяции:", population_variance,
      "Функцией:", population_variancef)

# Стандартное отклонение
standard_deviation = population_variance ** 0.5
standard_deviationf = np.std(npArray)
print("Стандартного отклонения:", standard_deviation,
      "Функцией:", standard_deviationf)

# Медиана
med = 0
if size % 2 == 0:
    mediana = data[size // 2]
else:
    mediana = (data[size // 2] + data[size // 2 + 1]) / 2
mediana_data = np.median(npArray)
print("Медиана:", mediana,
      "Функцией:", mediana_data)

# Мода
mode = max(set(data), key=data.count)
mode_data = stats.mode(data).mode
print("Мода:", mode,
      "Функцией:", mode_data)

# Коэффициент симметрии
skewness = 0
for x in data:
    skewness += ((x - avg) / standard_deviation) ** 3
skewness = skewness * size / ((size - 1) * (size - 2))
skewness_data = stats.skew(data)
print("Коэффициент симметрии:", skewness,
      "Функцией:", skewness_data)

# Квантили (25%, 50%, 75%)
first_quantile = data[size // 4] if size % 4 == 0 else (data[size // 4] + data[size // 4 + 1]) / 2
medium_quantile = mediana
last_quantile = data[3 * size // 4] if size % 4 != 0 else (data[3 * size // 4] + data[3 * size // 4 + 1]) / 2

quantile_25 = np.percentile(npArray, 25)
quantile_50 = np.percentile(npArray, 50)
quantile_75 = np.percentile(npArray, 75)
print("25% 50% 75%", first_quantile, medium_quantile, last_quantile,
      "\n25% 50% 75% Функцией", quantile_25, quantile_50, quantile_75)

# Интерквартильная ширина
interquartile = last_quantile - first_quantile
iqr_data = np.percentile(data, 75) - np.percentile(data, 25)
iqr_dataf = np.percentile(data, 75) - np.percentile(data, 25)
print("Интерквартильная ширина: ", iqr_data,
      "Функцией:", iqr_dataf)

# Графики
# Построение частотной гистограммы
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
count, bins, _ = plt.hist(data, bins=size//10, color='skyblue', edgecolor='black')
plt.xlim(bins[0], bins[-1])
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Частотная гистограмма')

# Построение вероятностной гистограммы с функцией плотности
plt.subplot(2, 2, 2)
count, bins, _ = plt.hist(data, bins=size//10, density=True, color='orange', edgecolor='black', alpha=0.7)
plt.xlim(bins[0], bins[-1])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(data), np.std(data))
plt.plot(x, p, 'k', linewidth=2)
plt.title('Вероятностная гистограмма с функцией плотности')

# Полигон распределения
plt.subplot(2, 2, 3)
plt.step(npArray, np.arange(1, len(npArray) + 1) / len(npArray))
plt.xlabel('Values')
plt.ylabel('Probability')
plt.title('Полигон')
plt.grid(True)

# Построение box plot
plt.subplot(2, 2, 4)
plt.boxplot(data, vert=True, patch_artist=True, showmeans=True, showfliers=True)
plt.title('Box Plot')
plt.xlabel('Значения')

# Нанесение нормального распределения на эмпирическую функцию
plt.figure(figsize=(12, 6))
plt.hist(data, bins=9, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Empirical Distribution')
plt.xlim(bins[0], bins[-1])

plt.plot(x, norm.pdf(x, mediana, standard_deviation), 'k', linewidth=2, label='Normal Distribution')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Empirical Distribution with Normal Distribution Overlay')


plt.show()
