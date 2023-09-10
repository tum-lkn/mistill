


def get_num(name: str) -> int:
    num = name.split('-')[1].lstrip('0')
    num = 0 if len(num) == 0 else int(num)
    return num


def get_pod(name: str, k: int) -> int:
    num = name.split('-')[1].lstrip('0')
    num = 0 if len(num) == 0 else int(num)
    if name.startswith('h-'):
        pod = int(num / (k * k / 4.))
    elif name.startswith('agg-') or name.startswith('tor-'):
        pod = int(num / (k / 2.))
    else:
        raise KeyError("Node {} does not belong to a pod.".format(name))
    return pod
