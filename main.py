import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def f(x, y):
    return y + np.cos(x / np.sqrt(3))

def euler_explicit(x0, y0, h, xf):
    n = int((xf - x0) / h)
    x = np.linspace(x0, xf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return x, y

def euler_cauchy(x0, y0, h, xf):
    n = int((xf - x0) / h)
    x = np.linspace(x0, xf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        y[i + 1] = y[i] + h * f(x[i] + h / 2, y[i] + (h / 2) * f(x[i], y[i]))

    return x, y

def improved_euler(x0, y0, h, xf):
    n = int((xf - x0) / h)
    x = np.linspace(x0, xf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        y_temp = y[i] + h * f(x[i], y[i])
        y[i + 1] = y[i] + (h / 2) * (f(x[i], y[i]) + f(x[i] + h, y_temp))

    return x, y

def runge_kutta_4(x0, y0, h, xf):
    n = int((xf - x0) / h)
    x = np.linspace(x0, xf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x, y

def exact_solution(x0, y0, xf):
    x = sp.symbols('x')
    y = sp.Function('y')(x)
    eq = sp.Eq(sp.Derivative(y, x), y + sp.cos(x / sp.sqrt(3)))
    sol = sp.dsolve(eq, y, ics={y.subs(x, x0): y0})
    x_values = np.linspace(x0, xf, 100)
    y_values = [sol.subs(x, val).rhs.evalf() for val in x_values]
    return x_values, y_values

def exact_solution_specific_x(x_values, y_values, specific_x_values):
    print("Точний розв'язок:")
    for x_val, y_val in zip(x_values, y_values):
        if round(x_val, 2) in specific_x_values:
            print(f"x = {x_val: .2f}, y = {y_val: .4f}")
    print()


# Початкові умови та параметри
x0 = 1.2
y0 = 2.1
xf = 2.2
h = 0.1

# Розв'язок методом Ейлера (явний)
x_euler_exp, y_euler_exp = euler_explicit(x0, y0, h, xf)

# Розв'язок методом Ейлера-Коші
x_euler_cauchy, y_euler_cauchy = euler_cauchy(x0, y0, h, xf)

# Розв'язок вдосконаленим методом Ейлера
x_improved_euler, y_improved_euler = improved_euler(x0, y0, h, xf)

# Розв'язок методом Рунге-Кутта 4-го порядку
x_rk4, y_rk4 = runge_kutta_4(x0, y0, h, xf)

# Розв'язок точного розв'язку
x_exact, y_exact = exact_solution(x0, y0, xf)

specific_x_values = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]  # Додайте сюди потрібні вам x-координати
exact_solution_specific_x(x_exact, y_exact, specific_x_values)



# Графічне відображення
plt.figure(figsize=(10, 6))
plt.plot(x_euler_exp, y_euler_exp, label='Метод Ейлера (явний)')
plt.plot(x_euler_cauchy, y_euler_cauchy, label='Метод Ейлера-Коші')
plt.plot(x_improved_euler, y_improved_euler, label='Вдосконалений метод Ейлера')
plt.plot(x_rk4, y_rk4, label='Метод Рунге-Кутта 4-го порядку')
plt.plot(x_exact, y_exact, label='Точний розв\'язок', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Порівняння чисельних методів з точним розв\'язком')
plt.legend()
plt.grid(True)
plt.show()


# Виведення значень функції для обчислених точок
print("Метод Ейлера (явний):")
for i in range(len(x_euler_exp)):
    print(f"x = {x_euler_exp[i]:.2f}, y = {y_euler_exp[i]:.4f}")

print("\nМетод Ейлера-Коші:")
for i in range(len(x_euler_cauchy)):
    print(f"x = {x_euler_cauchy[i]:.2f}, y = {y_euler_cauchy[i]:.4f}")

print("\nВдосконалений метод Ейлера:")
for i in range(len(x_improved_euler)):
    print(f"x = {x_improved_euler[i]:.2f}, y = {y_improved_euler[i]:.4f}")

print("\nМетод Рунге-Кутта 4-го порядку:")
for i in range(len(x_rk4)):
    print(f"x = {x_rk4[i]:.2f}, y = {y_rk4[i]:.4f}")

