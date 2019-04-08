"""
returns an empty list with shape (d0, d1, d2)
"""
def empty_list(d0, d1=None, d2=None):
    if d1 is None:
        return [[] for _ in range(d0)]
    elif d2 is None:
        return [[[] for _ in range(d1)] for __ in range(d0)]
    else:
        return [[[[] for _ in range(d2)] for __ in range(d1)] for ___ in range(d0)]
