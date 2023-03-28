import plotly.graph_objects as go
import numpy as np
import math
import pickle
import sys
import tensorflow as tf
from model.generator import read_fasta_as_one_hot_encoded

"""
sys.argv[1]: hamiltonian pkl path
sys.argv[2]: model
sys.argv[3]: fasta of sequence to plot
"""
grid_dataset = pickle.load(open(sys.argv[1], 'rb'))
x_min = np.min(grid_dataset[:, 0])
x_max = np.max(grid_dataset[:, 0])
y_min = np.min(grid_dataset[:, 1])
y_max = np.max(grid_dataset[:, 1])

size = int(math.sqrt(len(grid_dataset[:, 0])))
z = grid_dataset[:, 2].reshape(size, size)
x = grid_dataset[:, 0].reshape(size, size)
y = grid_dataset[:, 1].reshape(size, size)

vae = tf.keras.models.load_model(sys.argv[2])

ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(sys.argv[3]), tf.int8)
ds = ds.batch(1)
mu1, _, _ = vae.encoder.predict(ds)

ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(sys.argv[4]), tf.int8)
ds = ds.batch(1)
mu2, _, _ = vae.encoder.predict(ds)

ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(sys.argv[5]), tf.int8)
ds = ds.batch(1)
mu3, _, _ = vae.encoder.predict(ds)

ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(sys.argv[6]), tf.int8)
ds = ds.batch(1)
mu4, _, _ = vae.encoder.predict(ds)

ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(sys.argv[7]), tf.int8)
ds = ds.batch(1)
mu5, _, _ = vae.encoder.predict(ds)

# ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(sys.argv[8]), tf.int8)
# ds = ds.batch(1)
# mu6, _, _ = vae.encoder.predict(ds)

# get ham values for points
def get_zs(mu, grid_dataset):
    m,n = np.shape(mu)
    ham_z = np.zeros(m)
    for row_idx in range(0, m):
        dist = np.zeros(len(grid_dataset))
        for i in range(0, len(grid_dataset)):
            dist[i] = np.linalg.norm(mu[row_idx, :] - grid_dataset[i, 0:2])
        ham_z[row_idx] = grid_dataset[np.where(dist == np.min(dist))[0][0],2]+10
    return ham_z



# Camera angle
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-0.5, y=-2.5, z=1.5)
)

# plot
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y,colorbar=dict(title=u'Δ Hamiltonian'))]) # frames=frames

# wt BenM
fig.add_scatter3d(x=mu1[:,0], y=mu1[:,1],z =get_zs(mu1, grid_dataset), mode='markers',marker=dict(
            color='yellow',size=10), name="Squirrel WT")
# specific
fig.add_scatter3d(x=mu2[:,0], y=mu2[:,1],z =get_zs(mu2, grid_dataset), mode='markers',marker=dict(
            color='black',size=10), name="Rat WT")
# operational
fig.add_scatter3d(x=mu3[:,0], y=mu3[:,1],z =get_zs(mu3, grid_dataset), mode='markers',marker=dict(
            color='blue',size=10), name="Squirrel 5mut")
# # inversion
fig.add_scatter3d(x=mu4[:,0], y=mu4[:,1],z =get_zs(mu4, grid_dataset), mode='markers',marker=dict(
            color='green',size=10), name="Squirrel 6mut")
# dynamic
fig.add_scatter3d(x=mu5[:,0], y=mu5[:,1],z =get_zs(mu5, grid_dataset), mode='markers',marker=dict(
            color='red',size=10), name="Rat 6mut")


fig.update_scenes(xaxis_title_text='z0',
                  yaxis_title_text='z1',
                  zaxis_title_text=u'Δ Hamiltonian')
fig.update_layout(title='TRMP8 Local Landscape', autosize=True,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90),
                  legend=dict(
                      orientation="h",
                      yanchor="top",
                      y=1.0,
                      xanchor="right",
                      x=1),
                  scene_camera=camera
                  )

fig.write_image("output path")
# fig.show()
