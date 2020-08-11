from collections.abc import Iterable
import sys

import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from ipywidgets import widgets, interact

def sphere_points(sphere_density=100):
    theta = np.linspace(0, 2 * np.pi, sphere_density)
    phi = np.linspace(0, np.pi, sphere_density)
    x = np.outer(np.cos(theta), np.sin(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.ones(sphere_density), np.cos(phi))
    
    samples = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
    shape = (sphere_density, sphere_density)
    return samples, shape

def default_camera():
    camera = {
        "up": dict(x=0, y=0, z=1),
        "center": dict(x=0, y=0, z=0),
        "eye": dict(x=1.5, y=-1.5, z=1.5)
    }
    return camera

def default_scene(camera):
    return {
        "camera": camera,
        "xaxis": {
            "title": "",
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            # "autotick": True,
            "ticks": "",
            "showticklabels": False,
            "showbackground": False,
        },
        "yaxis": {
            "title": "",
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            # "autotick": True,
            "ticks": "",
            "showticklabels": False,
            "showbackground": False,
        },
        "zaxis": {
            "title": "",
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            # "autotick": True,
            "ticks": "",
            "showticklabels": False,
            "showbackground": False,
        },
    }

def default_layout(camera, plot_size, n_scenes=1):
    layout = {
        "width": plot_size[0],
        "height": plot_size[1],
        "autosize": False,
        "margin": dict(l=0, r=0, t=0, b=0),
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    }
    for scene_i in range(n_scenes):
        layout[f"scene{scene_i + 1}"] = default_scene(camera)
    return layout

def tonemap(value, exposure=2.2):
    return value
    # print(f"Non-finite pdfs: {np.count_nonzero(np.logical_not(np.isfinite(value)))} for array of size {value.shape}.")
    # value[np.logical_not(np.isfinite(value))] = 0
    # return np.power(value - np.min(value), 1 / exposure)

def plot_sphere(value_func, camera=default_camera(), plot_size=(300, 300), sphere_density=100, show=True):
    if not isinstance(value_func, Iterable):
        value_func = [value_func]
    n_funcs = len(value_func)
    
    subplots = make_subplots(rows=1, cols=n_funcs, specs=[[{"type": "surface", "is_3d": True}] * n_funcs])
    plot_size = (plot_size[0] * n_funcs, plot_size[1])
    
    samples, samples_shape = sphere_points(sphere_density)
    for func_i in range(n_funcs):  
        value = value_func[func_i](samples).reshape(samples_shape)
        subplots.append_trace(
            go.Surface(
                name="Sphere",
                x=samples[:, 0].reshape(samples_shape),
                y=samples[:, 1].reshape(samples_shape),
                z=samples[:, 2].reshape(samples_shape),
                surfacecolor=tonemap(value),
                text=value,
                colorscale="Cividis",
                showscale=False,
                scene=f"scene{func_i + 1}",
            ),
            row=1,
            col=func_i + 1
        )
    subplots.update_layout(default_layout(camera, plot_size, n_funcs))
    figure = go.FigureWidget(subplots)
    if show:
        init_notebook_mode(connected=True)
        iplot(figure, filename='jupyter-sphere')
    return figure
