from pylab import *
x = arange(0, 20, 0.1)
y = x**3 + 2*x + 2*cos(x)
plot(x, y)

xlabel('x')
ylabel('y')
title('Fonction: y = x^3 + 2x + 2*cos(x)')

grid()
show()