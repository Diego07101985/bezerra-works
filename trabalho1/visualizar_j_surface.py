# Para melhor entender a fun¸c˜ao de custo, vocˆe ir´a nessa parte do trabalho plotar
# valores da fun¸c˜ao de custo J sobre uma grade bidimensional de valores de θ0
# e de θ1

# Esses scripts geram um array bidimensional de
# valores de J(θ). Os valores gerados est˜ao contidos na faixa a seguir: −10 ≤
# θ0 ≤ +10 e −1 ≤ θ0 ≤ +4

theta0 = np.linspace(-10, 10)
theta1 = np.linspace(-1, 4)

# Um incremento de 0,01 ´e utilizado para gerar os
# valores de θ0 e de θ1.
learning_rate = 0.01

j_ts = np.zeros((len(theta0), len(theta1)))

food_truck = pe.load_food_truck()
x = food_truck['Profit in $ 10,000s']
y = food_truck['Population in City in 10,000s']

for ti0 in range(len(theta0)):
    for ti1 in range(len(theta1)):
        ts = [[theta0[ti0]], [theta1[ti1]]]
        j_ts[ti0,ti1] = custo_regrlin(theta0[ti0], theta1[ti1], x, y)

j_ts = np.transpose(j_ts)

fig = plt.figure(figsize=(10, 7))
ax = fig.gca(projection='3d')
theta0, theta1 = np.meshgrid(theta0, theta1)
surf = ax.plot_surface(theta0, theta1, j_ts, cmap=cm.coolwarm, rstride=2, cstride=2)
fig.colorbar(surf)
plt.xlabel('theta0')
plt.ylabel('theta1')





