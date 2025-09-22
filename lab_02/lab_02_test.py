import numpy as np
import plotly.graph_objects as go

R = 3
r = 1.5
k = 0.6
n = 5

u = np.linspace(0, 2 * np.pi, 200)
v = np.linspace(0, 4 * np.pi, 400)
u, v = np.meshgrid(u, v)

x = (R + r*np.cos(u) + k*np.sin(n*v)) * np.cos(v)
y = (R + r*np.cos(u) + k*np.sin(n*v)) * np.sin(v)
z = r * np.sin(u)

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

# Минимальные и максимальные координаты
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)
z_min, z_max = np.min(z), np.max(z)

print(f"X: min = {x_min:.12f}, max = {x_max:.12f}")
print(f"Y: min = {y_min:.12f}, max = {y_max:.12f}")
print(f"Z: min = {z_min:.12f}, max = {z_max:.12f}")
