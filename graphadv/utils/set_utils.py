import numpy as np
from graphgallery import asintarr


def union1d(arr1, arr2, toarr=False):
    arr1 = asintarr(arr1)
    arr2 = asintarr(arr2)
    arr = list(arr1.intersection(arr2))
    if toarr:
        arr = np.asarray(arr)
    return arr


def setdiff1d(arr1, arr2, toarr=False):
    arr1 = asintarr(arr1)
    arr2 = asintarr(arr2)
    arr = list(arr1.difference(arr2))
    if toarr:
        arr = np.asarray(arr)
    return arr
