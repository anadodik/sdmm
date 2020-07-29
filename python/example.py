import sys
sys.path.append('../build/lib')
import pysdmm as sdmm
from pysdmm import distributions as dist
from pysdmm import spaces
from pysdmm import opt
from pysdmm import RNG

import numpy as np

categorical = dist.Categorical()
categorical.pmf = np.arange(10)
categorical.prepare()
print(f"cdf={categorical.cdf}")

euclidian = spaces.EuclidianTangentSpace3()
euclidian.set_mean([[1, 1, 1], [2, 2, 2]])
inv_jacobians = np.zeros([2])
tangent = euclidian.to_tangent(np.ones([3]), inv_jacobians)
print(f"euclidian to tangent={tangent}")

directional = spaces.DirectionalTangentSpace()
directional.set_mean([[1, 0, 0], [0, 1, 0]])
inv_jacobians = np.zeros([2])
tangent = directional.to_tangent(np.array([0, 0, 1]), inv_jacobians)
print(f"directional to tangent={tangent}")

spatio_directional = spaces.SpatioDirectionalTangentSpace3()
spatio_directional.set_mean([[1, 1, 0, 0], [-1, 0, 1, 0]])
inv_jacobians = np.zeros([2])
tangent = spatio_directional.to_tangent(np.array([1, 0, 0, 1]), inv_jacobians)
print(f"spatio_directional to tangent={tangent}")

sdmm = dist.SDMM3()
sdmm.weight.pmf = np.arange(2) + 1
sdmm.tangent_space.set_mean([[1, 1, 0, 0], [-1, 0, 1, 0]])
sdmm.cov = [np.eye(3), np.eye(3)]
sdmm.prepare()

rng = RNG()
n_samples = 2
samples, inv_jacobians = sdmm.sample(rng, n_samples)
valid_samples = inv_jacobians != 0
samples = samples[valid_samples, :]
pdf = sdmm.pdf(samples)
weight = 1 / pdf

data = opt.SDMMData3()
data.reserve(samples.shape[0])
data.point = samples
data.weight = weight
data.heuristic_pdf = np.full((samples.shape[0],), -1)
data.size = n_samples

print(data.weight)

em = opt.SDMMEM3(2)
print(sdmm.weight.pmf)
sdmm = em.step(sdmm, data)
print(sdmm.cov)
