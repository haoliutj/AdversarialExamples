import numpy as np
import torch


# def add_perturbation(x,pert):
#     """
#     the perturbation add to burst format data in Website Finerprinting
#     must be keep the result increase in the original direction
#     e.g., x = [-2,3,-1], pert = [-0.03,-0.04,-0.1], x+pert=[-2-0.03,3+0,-1-0.1]=[-2.03,3,-1.1]
#
#     :param x: ndarray
#     :param pert: ndarray
#     :return: torch.Tensor
#     """
#
#     input_shape = x.shape
#
#     "Tensor to ndarray"
#     if type(x) == torch.Tensor:
#         x = x.data.cpu().numpy()
#     if type(pert) == torch.Tensor:
#         pert = pert.data.cpu().numpy()
#
#
#     result = []
#
#     for i in range(len(x)):
#         for j in range(len(x[i][0])):
#             a = x[i][0][j]
#             b = pert[i][0][j]
#             if a * b >= 0:
#                 temp = a + b
#                 result.append(temp)
#             else:
#                 temp = a
#                 result.append(temp)
#
#     result = torch.Tensor(np.array(result))
#     result = result.view(input_shape)
#
#     return result