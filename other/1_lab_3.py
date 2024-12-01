import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

# Считайте csv в DataFrame pandas при помощи pd.read_csv
iris = pd.read_csv("Iris.csv")
print(iris.head())

# Удалим ненужный столбец

iris = iris.drop(columns=['Id'])
print(iris.head())

# Проверим, сколько у нас разных видов ирисов 

print(iris.value_counts(['Species']))

# Проверим, связаны ли ширина и длина лепестков

sns.scatterplot(data=iris, x = "PetalWidthCm", y="PetalLengthCm", hue="Species")
plt.title('Length(Width)')
plt.show()

# Из графика видна линейная зависимоть между шириной и длинной лепестков ирисов,
# и видно, что виды ирисов отличаются друг от друга по ширине и длине лепестов 


# Попробуем для этой же цели другие виды графиков

sns.jointplot(data=iris, x = "PetalWidthCm", y="PetalLengthCm", hue="Species")
plt.show()

g= sns.FacetGrid(iris,col="Species")
g.map(sns.scatterplot,"PetalWidthCm","PetalLengthCm")
plt.show()

# Каждый из этих методов может быть использован для соотвественных специальных задач,
# например, scatterplot отображает сразу все данные и можно посмотреть на общую корреляцию,
# jointplot позволяет понять количество точек с определенным значением(распределение плотности), 
# с помощью facetgrip можно рассмотреть данные, отличающиеся по какому-то признаку, 
# отдельно друг от друга. Для данного случая мне кажется, что jointplot самый удобный метод


# Изучим распределение длины лепестков для каждого вида ирисов

sns.boxplot(data=iris, x="PetalLengthCm",y="Species",width =0.5)
plt.title('PentalLength(Species)')
plt.show()

sns.violinplot(data=iris, x="PetalLengthCm",y="Species")
plt.title('PentalLength(Species)')
plt.show()

# Посмотрим на все возможные для наших данных попарные зависимости
 
sns.pairplot(iris,hue="Species")
plt.show()

sns.pairplot(iris,hue="Species",diag_kind="hist")
plt.show()

# Вариант с гистограммами на диагонали мне кажется не таким удобным и понятным, как другой вариант
