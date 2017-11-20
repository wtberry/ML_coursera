import matplotlib.pyplot as plt
import numpy as np

sigmoid = lambda x: 1/(1+np.exp](-x)
x = linspace(-10, 10, 100)
y = linspace(-10, 10, 100)
plt.plot(x, sigmoid(x))
plt.grid()
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Sigmoid function')
plt.subtitle('Sigmoid')

plt.text(4,0.8,r'$\sigma(x)=\frac{1}{1+e^{-x}}$',fontsize=15)
plt.legend(loc='lower right')

plt.show()
