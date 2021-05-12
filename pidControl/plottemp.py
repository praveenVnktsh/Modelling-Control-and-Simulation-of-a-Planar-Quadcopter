import matplotlib.pyplot as plt
import json

with open('non.txt', 'r') as f:
    a = json.load(f)
with open('lin.txt', 'r') as f:
    b = json.load(f)

plt.plot(a, label = 'Non Linear Dynamics')
plt.plot(b, label = 'Linearized Dynamics')
plt.title('Failure of Linear Dynamics')
plt.grid()
plt.legend()
plt.show()

