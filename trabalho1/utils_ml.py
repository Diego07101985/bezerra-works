
def hypotheses(theta0, theta1, x_s):
    return theta0 + (theta1 * x_s)

def hypotheses_m(theta , x_s):
    return x_s.dot(theta) 