import numpy as np
import plotly.graph_objects as go

# Параметры фигуры
R = 3
r = 1.5
k = 0.6
n = 5

# Сетка параметров
u = np.linspace(0, 2 * np.pi, 200)
v = np.linspace(0, 4 * np.pi, 400)
u, v = np.meshgrid(u, v)

# Фигура
x = (R + r*np.cos(u) + k*np.sin(n*v)) * np.cos(v)
y = (R + r*np.cos(u) + k*np.sin(n*v)) * np.sin(v)
z = r * np.sin(u)

# Реальные границы (по данным из массива)
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)
z_min, z_max = np.min(z), np.max(z)

# Создание 3D поверхности
fig = go.Figure()

# Поверхность фигуры
fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale="Plasma", opacity=0.9))

# Параллелепипед (грани через линии)
# Координаты вершин параллелепипеда
corners = np.array([
    [x_min, y_min, z_min],
    [x_max, y_min, z_min],
    [x_max, y_max, z_min],
    [x_min, y_max, z_min],
    [x_min, y_min, z_max],
    [x_max, y_min, z_max],
    [x_max, y_max, z_max],
    [x_min, y_max, z_max]
])

# Рёбра параллелепипеда
edges = [
    (0,1),(1,2),(2,3),(3,0), # нижнее основание
    (4,5),(5,6),(6,7),(7,4), # верхнее основание
    (0,4),(1,5),(2,6),(3,7)  # вертикальные рёбра
]

for e in edges:
    fig.add_trace(go.Scatter3d(
        x=[corners[e[0],0], corners[e[1],0]],
        y=[corners[e[0],1], corners[e[1],1]],
        z=[corners[e[0],2], corners[e[1],2]],
        mode='lines',
        line=dict(color='black', width=3)
    ))

# Настройка сцены
fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    )
)

fig.show()
