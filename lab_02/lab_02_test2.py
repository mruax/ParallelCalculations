import numpy as np
import plotly.graph_objects as go

# Параметры фигуры
R = 3
r = 1.5
k = 0.6
n = 5

# Сетка параметров
u = np.linspace(0, 2*np.pi, 200)
v = np.linspace(0, 4*np.pi, 400)
u, v = np.meshgrid(u, v)

# Радиус в плоскости XY и z
r_xy = R + r*np.cos(u) + k*np.sin(n*v)
z_val = r*np.sin(u)

# Перевод в сферические координаты
rho = np.sqrt(r_xy**2 + z_val**2)
phi = np.arccos(z_val / rho)
theta = v

# Перевод обратно в декартовы через сферические
x = rho * np.cos(theta) * np.sin(phi)
y = rho * np.sin(theta) * np.sin(phi)
z = rho * np.cos(phi)  # совпадает с z_val

# Построение 3D поверхности
fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Plasma")])
fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    )
)
fig.show()
