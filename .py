# f = lambda num: ('Нечетное', 'Четное')[num % 2 == 0]

# print(f(12))

import numpy.random as npr
a = npr.randint(0, 100, size=100)
b, c = [], []
[b.append(el) if el & 1 else c.append(el) for el in a]

print(*b)