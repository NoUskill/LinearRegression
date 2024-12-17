# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

#Реальные коефиценты
class Data:
    def __init__(self,_seed,_num_samples):
        self.coefficients = [3.2, -1.5, 0.7, 2.8, -0.9]
        self.name_coef=['x1','x2','x3','x4','x5']
        self.name_result='y'
        self.num_samples=_num_samples
        self.data=self.DataGenerat(_seed)
    def DataGenerat(self,_seed):
        np.random.seed(_seed)
        x1 = np.random.uniform(0, 10, self.num_samples)
        x2 = np.random.uniform(0, 10, self.num_samples)
        x3 = np.random.uniform(0, 10, self.num_samples)
        x4 = np.random.uniform(0, 10, self.num_samples)
        x5 = np.random.uniform(0, 10, self.num_samples)
        y = self.coefficients[0]*x1 + self.coefficients[1]*x2 + self.coefficients[2]*x3 + self.coefficients[3]*x4 + self.coefficients[4]*x5
        data = pd.DataFrame({
        self.name_coef[0]: x1,
        self.name_coef[1]: x2,
        self.name_coef[2]: x3,
        self.name_coef[3]: x4,
        self.name_coef[4]: x5,
        self.name_result: y
        })
        return data




def Gradient_static(X,alpha=0.001,it=6000,epsilon=1e-10):
    iterations=it
    training_data=X.data[:int(X.num_samples*0.8)]
    test_data=X.data[int(X.num_samples*0.8):]
    training_coef_data=training_data[X.name_coef].values
    training_result=training_data[X.name_result].values
    test_coef_data =test_data[X.name_coef].values
    test_result = test_data[X.name_result].values
    coefficients_regression=np.zeros(len(X.name_coef))



    loss_0 = float('inf')
    for i in range(iterations):
        model_predict=np.dot(training_coef_data,coefficients_regression)
        error=model_predict-training_result
        grad=(2/len(training_result))*np.dot(training_coef_data.T,error)
        coefficients_regression=coefficients_regression - alpha*grad

        loss = (1 / (len(X.name_coef))) * np.sum(error ** 2)

        if abs(loss_0- loss)< epsilon:
            print(coefficients_regression,i)
            PaintGrafic(coefficients_regression, test_coef_data, test_result)
            return coefficients_regression
        loss_0 = loss


    print(coefficients_regression,iterations)
    PaintGrafic(coefficients_regression,test_coef_data,test_result)
    return coefficients_regression

def Gradient_Adagrad(X,alpha=10,max_iter=60000,epsilon=1e-10):
    iterations = max_iter
    training_data = X.data[:int(X.num_samples * 0.8)]
    test_data = X.data[int(X.num_samples * 0.8):]
    training_coef_data = training_data[X.name_coef].values
    training_result = training_data[X.name_result].values
    test_coef_data = test_data[X.name_coef].values
    test_result = test_data[X.name_result].values
    coefficients_regression = np.zeros(len(X.name_coef))+1
    G=np.zeros(len(X.name_coef))

    loss_0 = float('inf')
    for i in range(iterations):
        model_predict = np.dot(training_coef_data, coefficients_regression)
        grad = (2 / len(training_result)) * np.dot(training_coef_data.T, model_predict - training_result)
        G+=grad**2
        coefficients_regression -=alpha*grad/(np.sqrt(G)+epsilon)
        error = model_predict - training_result
        loss = (1 / (len(X.name_coef))) * np.sum(error ** 2)
        if abs(loss_0 - loss) < epsilon:
            print(coefficients_regression,i)
            PaintGrafic(coefficients_regression, test_coef_data, test_result)
            return coefficients_regression
        loss_0 = loss

    print(coefficients_regression,iterations)
    PaintGrafic(coefficients_regression, test_coef_data, test_result)
    return coefficients_regression

def Gradient_RMSProp(X,alpha=0.01,beta=0.9,max_iter=6000,epsilon=1e-10):
    iterations = max_iter
    training_data = X.data[:int(X.num_samples * 0.8)]
    test_data = X.data[int(X.num_samples * 0.8):]
    training_coef_data = training_data[X.name_coef].values
    training_result = training_data[X.name_result].values
    test_coef_data = test_data[X.name_coef].values
    test_result = test_data[X.name_result].values
    coefficients_regression = np.zeros(len(X.name_coef))
    E_g2 = np.zeros(len(X.name_coef))
    loss_0 = float('inf')
    for i in range(iterations):
        model_predict = np.dot(training_coef_data, coefficients_regression)
        grad = (2 / len(training_result)) * np.dot(training_coef_data.T, model_predict - training_result)
        E_g2=beta*E_g2+(1-beta)*grad**2
        coefficients_regression -= alpha * grad/(np.sqrt(E_g2)+epsilon)

        error = model_predict - training_result
        loss = (1 / (len(X.name_coef))) * np.sum(error ** 2)
        if abs(loss_0 - loss) < epsilon:
            print(coefficients_regression,i)
            PaintGrafic(coefficients_regression, test_coef_data, test_result)
            return coefficients_regression
        loss_0 = loss

    print(coefficients_regression,iterations)
    PaintGrafic(coefficients_regression, test_coef_data, test_result)
    return coefficients_regression


def Gradient_momentum(X,alpha=0.01,beta=0.9,max_iter=6000,epsilon=1e-10):
    iterations = max_iter
    training_data = X.data[:int(X.num_samples * 0.8)]
    test_data = X.data[int(X.num_samples * 0.8):]
    training_coef_data = training_data[X.name_coef].values
    training_result = training_data[X.name_result].values
    test_coef_data = test_data[X.name_coef].values
    test_result = test_data[X.name_result].values
    coefficients_regression = np.zeros(len(X.name_coef))
    m_exp=np.zeros(len(X.name_coef))

    loss_0 = float('inf')
    for i in range(iterations):
        model_predict = np.dot(training_coef_data, coefficients_regression)
        grad = (2 / len(training_result)) * np.dot(training_coef_data.T, model_predict - training_result)

        m_exp=m_exp*beta+(1-beta)*grad
        coefficients_regression-=alpha*m_exp

        error = model_predict - training_result
        loss = (1 / (len(X.name_coef))) * np.sum(error ** 2)

        if abs(loss_0 - loss) < epsilon:
                print(coefficients_regression,i)
                PaintGrafic(coefficients_regression, test_coef_data, test_result)
                return coefficients_regression
        loss_0 = loss
    print(coefficients_regression,iterations)
    PaintGrafic(coefficients_regression,test_coef_data,test_result)
    return coefficients_regression


def Matrix(X):
    training_data = X.data[:int(X.num_samples * 0.8)]
    test_data = X.data[int(X.num_samples * 0.8):]
    training_coef_data = training_data[X.name_coef].values
    training_result = training_data[X.name_result].values
    test_coef_data = test_data[X.name_coef].values
    test_result = test_data[X.name_result].values
    training_coef_data_T=training_coef_data.T

    try:
        coefficients_regression=np.linalg.inv(np.dot(training_coef_data_T,training_coef_data))@ training_coef_data_T @ training_result
    except np.linalg.LinAlgError:
        print("Решить матричным уравненем невозможноиспользуйте другой метод")
        coefficients_regression = np.zeros(len(X.name_coef))
        return coefficients_regression

    print(coefficients_regression)
    PaintGrafic(coefficients_regression, test_coef_data, test_result)
    return coefficients_regression



def PaintGrafic(coefficients_regression,test_coef_data,test_result):
    x = [i for i in range(len(test_result))]
    y_model = np.dot(test_coef_data, coefficients_regression).tolist()
    y = test_result
    x_new = np.linspace(min(x), max(x), 300)
    spline = make_interp_spline(x, y, k=3)  # k=3 для кубической интерполяции
    spline_model = make_interp_spline(x, y_model, k=3)  # k=3 для кубической интерполяции
    y_new = spline(x_new)
    y_new_model = spline_model(x_new)
    # Построение плавного графика
    plt.plot(x_new, y_new, color='blue', label='Корректные значения')
    plt.plot(x_new, y_new_model, color='green', label='Предсказанные значения')
    plt.scatter(x, y, color='red', s=2, label='Исходные данные')
    # Добавление заголовка и меток осей
    plt.title('Плавный график с интерполяцией')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()




arr=Data(25,150)
print("Real coef:",arr.coefficients)
Gradient_static(arr)
Gradient_RMSProp(arr)
Gradient_momentum(arr)
Gradient_Adagrad(arr)
Matrix(arr)