import numpy as np
import time

def raising_operator2(j):
    ''' Creates the angular momentum raising operator for j

    In the j,mj basis running from max(mj) to min (mj) creates a matrix that represents the operator j+\|j,mj> = \|j,mj+1>

    Args:
        j (float) : value of the angular momentum

    Returns:
        j+ (numpy.ndarray) : Array representing the operator J+, has shape ((2j+1),(2j+1))

    '''
    assert float(2*j+1).is_integer()

    mj_list = np.arange(-j, j)
    elements = np.sqrt(j*(j+1)-mj_list*(mj_list+1))
    jplus = np.diag(elements, 1)

    return jplus

t0 = time.time()
for _ in range(1000):
    a = raising_operator2(12)
print(time.time() - t0)

# print(a)