import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1.1
#
# A - "шахматную" из нулей и единиц, размера 6 на 3, левое верхнее значение (A[0][0]) равно 1,

A = np.array([[0,1,0],
              [1,0,1],
              [0,1,0],
              [1,0,1],
              [0,1,0],
              [1,0,1]]
              )

# В - из чисел от 1 до 24, записанных в виде змейки, размера 6 на 4,
B = np.array([[1,2,3,4],
              [8,7,6,5],
              [9,10,11,12],
              [16,15,14,13],
              [17,18,19,20],
              [24,23,22,21]])

#C - из случайных целых чисел от 2 до 10 (обе границы включительно), размера 4 на 3
C = np.random.randint(2,10,(4,3))

#D - из нулей с единичками на главной диагонали, размера 4 на 4.

D = np.diag([1,1,1,1])

# Создайте из этих матриц "лоскутную" матрицу S

S = np.vstack([np.hstack([A,B]),np.hstack([C,D])])

# После этого допишите к полученной матрице S матрицу F размера 10 на 2 из нулей, чтобы получилась матрица G:

G = np.hstack([S,np.zeros((10,2),dtype = int)])

# 1.2
#
# Реализуйте функцию, принимающую на вход матрицу X и 
# некоторое число a и возвращающую ближайший к числу элемент матрицы

def find_nearest_neighbour(X, a):
    X1 = X-a
    index = (X1**2).argmin() 
    return X.reshape((1,X.size))[0][index]

# 1.3
#
# Очень странная нейросеть

A = np.array([[1. ,2. ,3. ,4. ,5.],
              [6. ,7. ,8. ,9. ,10.],
              [11. ,12. ,13. ,14. ,15.],
              [16. ,17. ,18. ,19. ,20.],
              [21. ,22. ,23. ,24. ,25.]])
X = np.array([2., 2., 2. , 4. , 4. ])
B = np.array([-3., -2., -1. , 0. , 1. ])


def very_strange_neural_network(A, B, X):
    A2 = A*A.transpose()
    return np.dot(np.dot(X,A2),B.transpose())


print(very_strange_neural_network(A, B, X))

# 1.4
#
# Джунгли зовут

def find_deep_sea_area(M):
    num = 0
    for x in M:
        for y in x:
            if y<-5:
                num+=1
    return num            

def find_water_volume(M):
    num = 0
    for x in M:
        for y in x:
            if y<0:
                num -= y
    return num

def find_max_height(M):
    max = M[0][0]
    for x in M:
        for y in x:
            if y>max:
                max = y
    return max

# Можно подставить свой пример
M = np.array([
    [-7, -3, -1, 0],
    [-4, -3, 1, 19],
    [-2, 0, 4, 25],
    [-1, 3, 6, 9]
])

# простая проверка для примера выше
assert np.isclose(find_deep_sea_area(M), 1)
assert np.isclose(find_water_volume(M), 21)
assert np.isclose(find_max_height(M), 25)

print("Общая площадь моря на карте -", find_deep_sea_area(M), "м^2")
print("Общий объем воды на карте -", find_water_volume(M), "м^3")
print("Максимальный уровень над уровнем моря на карте -", find_max_height(M), "м")

# 1.5
#
# Острова сокровищ

def count_all_islands(a):
    s = np.sum(np.diff(a)**2)
    return s - s//2

# можно подставить свой пример

a = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1])

# простая проверка для примера выше
assert count_all_islands(a) == 4
print(count_all_islands(a))

# 1.6
# 
# Маскарад

def swap_mask_for_average(X, a):
    av = np.sum(X)/X.size
    return np.where(X<=5, X, av) # TODO

# Можно подставить свой пример
M = np.array([
    [-7, -3, -1, 0],
    [-4, -3, 1, 19],
    [-2, 0, 4, 25],
    [-1, 3, 6, 9]
])
a = 5

print(swap_mask_for_average(M, a))


# 1.7
#
# По горячим трейсам

def count_trace_diff(M):
    return np.trace(M) - np.trace(np.flip(M,0))

# Можно подставить свой пример
M = np.array([
    [-7, -3, -1, 0],
    [-4, -3, 1, 19],
    [-2, 0, 4, 25],
    [-1, 3, 6, 9]
])

# простая проверка для примера выше
assert np.allclose(count_trace_diff(M), 3)

count_trace_diff(M)

# 1.8 
#
# Царь-гора

def create_mountain(a):
    a = np.sort(a)
    b=a
    i=a.size
    while i>0:
        a = a+1
        b = np.vstack([b,a])
        i -=1
    b = np.hstack([b,np.flip(b,1)])
    b = np.vstack([b, np.flip(b)])    
    return b



# Можно подставить свой пример
a = np.array([0, 1, 2, 3, 4])

create_mountain(a)

# 1.9
#
# Монохромная фотография

def custom_blur(P, C):
    (str,col) = P.shape
    R = np.ones((str-C+1,col-C+1))
    i = 0
    while i < str-C+1:
        j=0
        while j < col-C+1:
            R[i,j] = np.sum(P[i:i+C,j:j+C]) / (C**2)
            j+=1
        i +=1
    return R


# можно подставить свой пример
P = np.arange(0, 12).reshape((3, 4))
kernel = 2

# простая проверка для примера выше
assert np.allclose(custom_blur(P, kernel),
                   np.array([[2.5, 3.5, 4.5], [6.5, 7.5, 8.5]]))
custom_blur(P, 2 )

# 1.10
#
# Функция проверки

def check_successful_broadcast(*matrices):
    prev = matrices[0]
    state = True
    for cur in matrices:
        min = cur
        max = prev
        if(len(cur) > len(prev)):
            min = prev
            max = cur

        i = len(max)
        j = len(min)


        while j>0:
            if (max[-j] == min[-j]  or  max[-j] == 1 or min[-j] == 1):
                pass
            else :
                state = False
                return False
            j-=1


        prev = cur    
    return True    
            


assert check_successful_broadcast((5, 6, 7), (6, 7), (1, 7))
# можно ещё потестировать на своих примерах

check_successful_broadcast((5, 6, 7), (6, 7), (1, 7))

# 1.11
#
# Попарные расстояния

