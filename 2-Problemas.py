# Vector X de 100 componentes IMPAR

import numpy as np 

x = np.arange(1,200.1,2) # 100 elementos
y = x*np.sin(x)/(x+1)

z = sum(y[-21:]) # Suma desde el elemento 21 (de atrás hacia delante) hasta el último elemento

print(z)

# Capacidad calorífica molar de un sólido

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

e = 40.9141 
N = 5
k = 0.5817

#a)

T = np.arange(315,415.1,10) # Temperaturas
cv = 3*N*k*(e/(k*T))**2*(np.e**(e/(k*T)))/((np.e**(e/(k*T))-1)**2)

print(f"T = {T}\n\nCv(T) = {cv}\n")

#b)

plt.plot(T,cv,'or')
plt.title("Cv(T)")
plt.legend("Cv(T)")
plt.xlabel("T")
plt.ylabel("Cv")
plt.show()

# Utilizar el comando linalg.solve para resolver el sistema

import numpy as np

ecu_xy = np.array([[1,-2],[2,1]]) # [x,y]
sol_ecu_xy = np.array([5,11]) 

solucion = np.linalg.solve(ecu_xy,sol_ecu_xy)
print(solucion)

# Ejercicio transbordador espacial

import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

u = 1800
m0 = 1.6 * 10**5
q = 2600
G = 9.81 # m/s^2

#a)

t = np.arange(10,30.1,0.5) 

v = u*np.log(m0/(m0-q*t))-G*t

datos_tbl_v = {"t":t, "v":v}
tbl_v = pd.DataFrame(datos_tbl_v)
display(tbl_v)

#b)

plt.title("Gráfica: Velocidad")
plt.plot(t,v, "--g")
plt.xlabel("Eje x: Tiempo")
plt.ylabel("Eje y: Velocidad")
plt.legend(["V"])

plt.grid(True)  # Agregar cuadrícula
plt.show()

# Script que calcule la sumatoria S(n)

import matplotlib.pyplot as plt

n = 15 

def Calcular_Sn(n):
    res = sum((-1)**k/(2*k+1) for k in range(n+1))
    return res

res_Sn = Calcular_Sn(n)

print(f'S(n = {n}) = {res_Sn}')

valores_sn = [(n, Calcular_Sn(n)) for n in range(16)]

with open('sn.txt', 'w') as fichero:
    fichero.writelines([f'n={n}, S(n)={sn}\n' for n, sn in valores_sn])

n_val, sn_val = zip(*valores_sn)
plt.plot(n_val, sn_val,'-r')
plt.xlabel('n')
plt.ylabel('S(n)')
plt.title('S(n) frente a n')
plt.show()

# Vector X de 200 componentes IMPAR

import numpy as np 

x = np.arange(1,400.1,2) # 200 elementos
y = x*np.sin(x)/(x+1)

z = sum(y[-22:]) # Suma desde el elemento 22 (de atrás hacia delante) hasta el último elemento

print(z)

# Partiendo del vector [3,7] construir una sucesión

import matplotlib.pyplot as plt

x = [3,7]

while x[-1] < 1000:
    suces = x[-1]+x[-2]
    x.append(suces)
    
plt.plot(range(len(x)),x,'-o')
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.title("Componentes del vector x")
plt.show()

# Utilizar el comando diff para calcular la derivada

import sympy as sp

x = sp.Symbol('x')
func = sp.log(sp.sin(x**2 + 1))

func_drv = sp.diff(f, x)
print(func_drv)

# Número de Reynolds

import numpy as np
import matplotlib.pyplot as plt

d = 1
u = 2
L = 8
mu = 0.4

arr1 = [] # Valores de u
arr2 = [] # Valores de Re(u)

Re = lambda d,u,L,mu : (d*u*L)/mu

while(Re(d,u,L,mu) <= 70):
    arr2.append(Re(d,u,L,mu))
    arr1.append(u)
    u += 0.2
    
plt.plot(arr1,arr2,'-or')
plt.xlabel("u")
plt.ylabel("Re")
plt.title("Re(u)")
plt.show()

# Valor contenido en la variable x

i=1
x=5
while x>10:
    x+=2
    i+=1
print(x)

# Valor contenido en la variable s

A=[1,3,6,10]
s=0
for a in A:
    s = s + a
print(s)

# Vector X de 200 componentes PAR

import numpy as np 

x = np.arange(2,400.1,2) # 200 elementos
y = x*np.sin(x)/(x+3)

z = sum(y[-21:]) # Suma desde el elemento 21 (de atrás hacia delante) hasta el último elemento

print(z)

# Constante de reacción química

import numpy as np
import matplotlib.pyplot as plt


a = 28
b = 12
c = 14

#a)

x = np.arange(0,2.1,0.1) # Temperaturas
K = (c+x)/((a-2*x)**2*(b-x))

datos_tbl_v = {"x":x, "K":K}
tbl_v = pd.DataFrame(datos_tbl_v)
display(tbl_v)

#b)

plt.plot(x,K,'ob')
plt.title("K(x)")
plt.legend("K(x)")
plt.xlabel("x")
plt.ylabel("K(x)")
plt.show()