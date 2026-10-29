import cudaq
@cudaq.kernel
def dmrg_mult6(q: cudaq.qvector):
    ry(3.141592653589793, q[0])
    ry(3.141592653589793, q[2])
    ry(3.141592653589793, q[4])
    ry(3.141592653589793, q[6])
    ry(3.141592653589793, q[8])
