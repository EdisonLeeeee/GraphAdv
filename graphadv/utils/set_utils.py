import numpy as np


def union1d(arr1, arr2, to_arr=False):
    arr1 = check_and_convert_set(arr1)
    arr2 = check_and_convert_set(arr2)
    arr = list(arr1.intersection(arr2))
    if to_arr:
        arr = np.asarray(arr)
    return arr


def setdiff1d(arr1, arr2, to_arr=False):
    arr1 = check_and_convert_set(arr1)
    arr2 = check_and_convert_set(arr2)
    arr = list(arr1.difference(arr2))
    if to_arr:
        arr = np.asarray(arr)
    return arr
