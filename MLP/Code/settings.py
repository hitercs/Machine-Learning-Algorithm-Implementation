#-*- encoding: utf-8 -*-
import numpy as np
read_buffer_size = 100000
write_buffer_size = 100000
data_gen_parameters = [
    {
        "mean": np.zeros(3),
        "cov": np.eye(3)
    },
    {
        "mean": np.array([0, 1, 0]),
        "cov" : np.array([[1, 0, 1], [0, 2, 2], [1, 2, 5]])
    },
    {
        "mean": np.array([-1, 0, 1]),
        "cov": np.array([[2, 0, 0], [0, 6, 0], [0, 0, 1]])
    },
    {
        "mean": np.array([0, 0.5, 1]),
        "cov": np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]])
    }
]
# Fake DataGen Setting
# data_gen_parameters = [
#     {
#         "mean": np.zeros(3),
#         "cov": np.eye(3)
#     },
#     {
#         "mean": np.array([100, 100, 100]),
#         "cov" : np.array([[1, 0, 1], [0, 2, 2], [1, 2, 5]])
#     },
#     {
#         "mean": np.array([-100, -100, -100]),
#         "cov": np.array([[2, 0, 0], [0, 6, 0], [0, 0, 1]])
#     },
#     {
#         "mean": np.array([500, 500, 500]),
#         "cov": np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]])
#     }
# ]
seed = 1234
init_variance = 1
epsilon = 1E-6
test_samples = np.array([[0, 0, 0],
    [-1, 0, 1],
    [0.5, -0.5, 0],
    [-1, 0, 0],
    [0, 0, -1]
])