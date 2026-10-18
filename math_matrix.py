import math as mth
import random as rnd

def matrix_aritc(x, y, function):
    if not isinstance(x, (int, float, complex, list)):
        return None
    if not isinstance(x, list):
        if function == '+':
            return x + y
        elif function == '-':
            return x - y
        elif function == '*':
            return x * y
        elif function == '/':
            return x / y
        elif function == '%':
            return x % y
        elif function == '**':
            return x ** y
        elif function == '//':
            return x ** (1 / y)
        elif function == 'log':
            return mth.log(x, y)
        elif function == 'gcd':
            return mth.gcd(x, y)
        elif function == 'lcm':
            return mth.lcm(x, y)
    else:
        A = []
        for i in range(min(len(x), len(y))):
            A.append(matrix_aritc(x[i], y[i], function))
        return A

def matrix_function(x, params, function):
    if not isinstance(x, (int, float, complex, list)):
        return None
    if not isinstance(x, list):
        if function == 'x':
            return x
        if function == '+':
            return x + params
        elif function == '-':
            return x - params
        elif function == '*':
            return x * params
        elif function == '/':
            return x / params
        elif function == '%':
            return x % params
        elif function == '**':
            return x ** params
        elif function == '//':
            return x ** (1 / params)
        elif function == 'log':
            return mth.log(x, params)
        elif function == 'gcd':
            return mth.gcd(x, params)
        elif function == 'lcm':
            return mth.lcm(x, params)
        elif function == 'leaky relu':
            return max(x, x * max(min(params, 1), 0))
        elif function == 'acos':
            return mth.acos(x)
        elif function == 'acosh':
            return mth.acosh(x)
        elif function == 'asin':
            return mth.asin(x)
        elif function == 'asinh':
            return mth.asinh(x)
        elif function == 'atan':
            return mth.atan(x)
        elif function == 'atanh':
            return mth.atanh(x)
        elif function == 'abs':
            return abs(x)
        elif function == 'ceil':
            return mth.ceil(x)
        elif function == 'cos':
            return mth.cos(x)
        elif function == 'cosh':
            return mth.cosh(x)
        elif function == 'clamp':
            return max(min(x, params[1]), params[0])
        elif function == 'erf':
            return mth.erf(x)
        elif function == 'erfc':
            return mth.erfc(x)
        elif function == 'exp':
            return mth.exp(x)
        elif function == 'elu':
            return mth.exp(x) - 1 if x < 0 else x
        elif function == 'floor':
            return mth.floor(x)
        elif function == 'gamma':
            return mth.gamma(x)
        elif function == 'gelu':
            return 0.5 * x * (1 + mth.erf(x / mth.sqrt(2)))
        elif function == 'mish':
            return x * mth.tanh(mth.log(1 + mth.exp(x)))
        elif function == 'round':
            return round(x)
        elif function == 'sin':
            return mth.sin(x)
        elif function == 'sinh':
            return mth.sinh(x)
        elif function == 'sigmoid':
            return 1 / (1 + mth.exp(-x))
        elif function == 'step':
            return round(max(min(x, 1), 0))
        elif function == 'selu':
            return params[1] * ((params[0] * mth.exp(x)) - params[0] if x <= 0 else x)
        elif function == 'softplus':
            return mth.log(1 + mth.exp(x))
        elif function == 'swish':
            return x / (1 + mth.exp(-x))
        elif function == 'step':
            if params[3] == 'ceil':
                return params[1] if x >= params[2] else params[0]
            elif params[3] == 'floor':
                return params[1] if x > params[2] else params[0]
            elif params[3] == 'netral':
                return (params[1] if x > params[2] else params[0]) if x != params[2] else 0.5
        elif function == 'tan':
            return mth.tan(x)
        elif function == 'tanh':
            return mth.tanh(x)
        elif function == 'trunc':
            return mth.trunc(x)
        elif function == 'relu':
            return max(x, 0)
    else:
        A = []
        for i in range(len(x)):
            A.append(matrix_function(x[i], params, function))
        return A

def matrix_sum_prod(x, function):
    if not isinstance(x, (int, float, complex, list)):
        return None
    if not isinstance(x, list):
        return x
    
    if function == 'sum':
        A = 0
        for i in range(len(x)):
            A += matrix_sum_prod(x[i], function)
        return A
    elif function == 'prod':
        A = 1
        for i in range(len(x)):
            A *= matrix_sum_prod(x[i], function)
        return A

def matrix_flatten(x):
    if not isinstance(x, list):
        return [x]
    
    A = []
    for i in range(len(x)):
        A.extend(matrix_flatten(x[i]))
    return A

def matrix_M_function(x, function):
    if function == 'mean':
        return sum(x) / len(x)
    elif function == 'median':
        sorted_x = sorted(x)
        n = len(sorted_x)
        mid = n // 2
        if n % 2 == 1:
            return sorted_x[mid]
        else:
            return (sorted_x[mid - 1] + sorted_x[mid]) / 2
    elif function == 'modus':
        modus_data = {}
        modus_value = None
        for i in x:
            if modus_data.get(i) == None:
                modus_data[i] = 0
            modus_data[i] += 1

            if modus_value == None:
                modus_value = i
            elif modus_data[i] > modus_data[modus_value]:
                modus_value = i
        return modus_value

def matrix_generate_list_random(x, function, params):
    new_list = []
    for i in range(x):
        if function == 'random':
            new_list.append(rnd.random())
        elif function == 'randint':
            new_list.append(rnd.randint(params[0], params[1]))
        elif function == 'uniform':
            new_list.append(rnd.uniform(params[0], params[1]))
        elif function == 'choice':
            new_list.append(rnd.choice(params))
        elif function == 'choices':
            new_list.append(rnd.choices(params[0], k = params[1]))
    return new_list

def matrix_dot_prod(x,y):
    return matrix_sum_prod(matrix_aritc(x, y, '*'), 'sum')

def matrix_list_nestlist(x, y, z, function, block):
    A = []
    if block == 'aritc':
        for i in range(z):
            A.append(matrix_aritc(x, y[i], function))
    elif block == 'function':
        for i in range(z):
            A.append(matrix_function(x[i], y, function))
    elif block == 'sum / prod':
        for i in range(z):
            A.append(matrix_sum_prod(x, y[i], function))
    elif block == 'dot prod':
        for i in range(z):
            A.append(matrix_dot_prod(x, y[i]))
    return A

def matrix_max_min(x, function):
    if not isinstance(x, (int, float, complex, list)):
        return None
    if not isinstance(x, list):
        return x

    A = []
    for i in x:
        A.append(matrix_max_min(i, function))
    return max(A) if function == 'max' else min(A)
