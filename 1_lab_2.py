import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# чтение данных из файла

data = pd.read_csv('data_numpy_lab.csv')
print(data.head())

# Перевод в СИ 

data["mass, gramm"] = data["mass, gramm"]*1e-3
data["length, cm"] = data["length, cm"]*1e-2

# Переименовывание колонок

data = data.rename(columns={"Unnamed: 0":"","t, sec":"t", "sigma_t, sec":"sigma_t", "mass, gramm":"mass", "length, cm":"length", "phi, rad":"phi"})


print(data.head())

assert data.mass.mean() < 0.3
assert np.allclose(data.length.mean(), 1.155)
assert all(' ' not in column for column in data.columns)

# Добавление новых колонок со значениями 

def f_omega(row):
    return 2*np.pi*row["N"]/row["t"]

data["omega"] = data.apply(f_omega,axis=1)

def f_sigma_omega(row):
    return row["omega"]/row["t"]*row["sigma_t"]

data["sigma_omega"] = data.apply(f_sigma_omega,axis=1)

def f_omega_down(row):
    return row["phi"]/row["t"]

data["omega_down"] = data.apply(f_omega_down,axis=1)

def f_sigma_down(row):
    return row["omega_down"]*row["sigma_t"]/row["t"]

data["sigma_down"] = data.apply(f_sigma_down,axis=1)

g=9.8

def f_momentum(row):
    return row["mass"]*g*row["length"]

data["momentum"] = data.apply(f_momentum,axis=1)

def f_momentum_down(row):
    return row["mass"]*row["phi"]*(row["length"]/row["t"])**2

data["momentum_down"] = data.apply(f_momentum_down,axis=1)

def f_sigma_momentum(row):
    return row["momentum_down"]*2*row["sigma_t"]/row["t"]

data["sigma_momentum"] = data.apply(f_sigma_momentum,axis=1)


assert np.allclose(data.momentum_down.iloc[0], 5.892e-07)
assert np.allclose(data.sigma_omega[0:5], 3.5e-04, atol=3e-5)
assert np.allclose(data.sigma_momentum[0:5], 4.4e-09, atol=1e-9)

print(data.head())

# Cредние значения колонок 
# omega, sigma_omega, momentum иmomentum_down 
# для каждой уникальной массы

grouped_data = data[["mass","omega","sigma_omega","omega_down","sigma_down","momentum","sigma_momentum","momentum_down"]].groupby(["mass"]).mean()


assert 0.273 in grouped_data.index
assert np.allclose(grouped_data.omega[0.273], 0.1433)

print(grouped_data)

#  построить график зависимости  Ω(M) 
#  угловой скорости от момента инерции

omega_np = np.array(grouped_data.omega)
momentum_np = np.array(grouped_data.momentum)

x= momentum_np
y = omega_np

# Воспользуйтесь np.polyfit
coefs = np.polyfit(x,y,1)

# Чтобы прямая построилась красиво, немножко заходя за точки
x_lsq = np.linspace(momentum_np.min() * 0.5, momentum_np.max() * 1.1, 100)

# Примените np.polyval к коэффициентам и x_lsq
y_lsq = np.polyval(coefs,x_lsq)

fig = plt.figure(figsize=(12, 8))

plt.plot(x_lsq,y_lsq,'r-',label="y={:.2f}+{:.2f}x".format(coefs[1],coefs[0]))
plt.plot(x,y,'b.')
plt.title("\u03A9"+"(M)",fontsize=14)
plt.xlabel("Момент инерции, M",fontsize =12)
plt.ylabel("Угловая скорость, \u03A9", fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Предположим, что погрешность увеличилась

grouped_data['sigma_down'] *= 10
grouped_data['sigma_momentum'] *= 10

# Отобразим на графике

omega_down_np = np.array(grouped_data.omega_down)
momentum_down_np = np.array(grouped_data.momentum_down)

x = momentum_down_np
y = omega_down_np

# Снова polyfit, но с дополнительным параметром и возвращающий ковариацию!
coefs, cov = np.polyfit(x,y,1,cov=True)

# Чтобы прямая построилась снова красиво
x_lsq = np.linspace(momentum_down_np.min() * 0.3, momentum_down_np.max() * 1.1, 100)

# Посчитайте корень диагональных элементов, должен получиться массив размером (2,)
lsq_stds = np.array([cov[0][0]**0.5,cov[1][1]**0.5])

# Знакомый polyfit, но три раза
y_lsq = np.polyval(coefs,x_lsq)
y_lsq_lower = np.polyval(coefs-lsq_stds,x_lsq)
y_lsq_upper = np.polyval(coefs+lsq_stds,x_lsq)

fig = plt.figure(figsize=(12, 8))

# YOUR CODE HERE
plt.errorbar(x,y, xerr = np.array(grouped_data.sigma_momentum),yerr = np.array(grouped_data.sigma_down),fmt=".",ecolor="green")
plt.plot(x_lsq, y_lsq,'r')
plt.plot(x_lsq, y_lsq_lower,'r')
plt.plot(x_lsq, y_lsq_upper,'r')
plt.fill_between(x_lsq, y_lsq, y_lsq_lower,alpha=0.2)
plt.fill_between(x_lsq, y_lsq, y_lsq_upper,alpha=0.2)
plt.title("\u03A9"+" down(M down)",fontsize=14)
plt.xlabel("Момент инерции, M down",fontsize =12)
plt.ylabel("Угловая скорость, \u03A9 down", fontsize=12)
plt.legend()
plt.grid()
plt.show()
plt.show()
