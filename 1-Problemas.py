# Ejercicio - Energía Potencial (Ep)

import numpy as np
import pandas as pd

k = np.arange(1,10.1,1)
x = np.arange(5,7.1,0.5)

Ep = 0.5*k[:,np.newaxis]*x**2

tbl_Ep = pd.DataFrame(Ep, index=k, columns=x)

tbl_Ep.columns.name = "x"
tbl_Ep.index.name = "k"

display(tbl_Ep)

# Ejercicio - Darcy-Weisbach

import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f = 0.0324
Q = 2.5 
L = 0.5
G = 9.81

#a)

D = np.arange(0.5,3.1,0.25)

hf = (8*f/(G*m.pi**2*D**5))*Q**2*L

datos_tbl_hf = {"D":D, "hf (D)":hf}
tbl_hf = pd.DataFrame(datos_tbl_hf)
display(tbl_hf)

#b)

plt.plot(D,hf,"*r")

plt.title("Gráfica: hf (D)")
plt.xlabel("D")
plt.ylabel("hf (D)")
plt.legend(["hf (D)"])

plt.grid(True)  # Agregar cuadrícula

plt.show()

# Vector X de 100 componentes

x = np.arange(2,200.1,2) # 100 elementos
y = x*np.sin(x)/(x+1)

z = sum(y[-21:]) # Suma desde el elemento 21 (de atrás hacia delante) hasta el último elemento

print(z)

# Fracciones

from fractions import Fraction

frac = (Fraction(2,3)+Fraction(3,13))**2

print(frac)

import sympy as s

func = (2*m.pi**2)/(1+m.e**m.sqrt(3))

ans = s.N(func, 20)

print(ans)

# Ejercicio Ley Logística

import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N0 = 100
r = 0.19
K = 10e4

#a)

t = np.arange(0,10.1,1) # Se imprime la hora cero solo para fines de confirmación NO = 100

Nt = (N0*K)/(N0+(K-N0)*m.e**(-r*t))

datos_tbl_Nt = {"Tiempo (horas)":t, "N(t)":Nt}
tbl_Nt = pd.DataFrame(datos_tbl_Nt)
display(tbl_Nt)

#b)

plt.title("Gráfica: N(t)")

plt.plot(t,Nt)
plt.xlabel("Tiempo")
plt.ylabel("Número de invidiuos")
plt.legend(["N(t)"])

plt.grid(True)  # Agregar cuadrícula

plt.show()

# Ejercicio transbordador espacial

import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

u = 1800
m0 = 1.6 * 10**5
q = 2600
G = 9.81

#a)

t = np.arange(10,30.1,0.5) 

v = u*np.log(m0/(m0-q*t))-G*t

datos_tbl_v = {"t":t, "v":v}
tbl_v = pd.DataFrame(datos_tbl_v)
display(tbl_v)

#b)

plt.title("Gráfica: velocidad")
plt.plot(t,v, "--g")
plt.xlabel("Eje x: Tiempo")
plt.ylabel("Eje y: velocidad")
plt.legend(["v"])

plt.grid(True)  # Agregar cuadrícula
plt.show()

# Ejercicio - Energía Cinética (Ec)

import numpy as np
import pandas as pd

m = np.arange(1,5.1,1)
v = np.arange(0,1.1,0.1)

Ec = (m*v[:,np.newaxis]**2)/2

tbl_Ec = pd.DataFrame(Ec, index=v, columns=m)

tbl_Ec.columns.name = "m"
tbl_Ec.index.name = "v"

display(tbl_Ec)

# Región matriz

matriz_M = np.array((["a","b","c","d","e"],["f","g","h","i","j"],["k","l","m","n","o"],["p","q","r","s","t"],["u","v","w","x","y"]))

print("Matriz M original:","\n",matriz_M,"\n")

print("Escoger el comando que selecciona la región indicada", matriz_M[1:4,1:3],"\n")