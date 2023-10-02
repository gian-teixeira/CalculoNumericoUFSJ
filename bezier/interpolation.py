import numpy as np
import matplotlib.pyplot as plt
import linear

def bezier_interpolate(points):
    n = len(points) - 1

    # Coefficients matrix
    C = 4 * np.eye(n)
    np.fill_diagonal(C[1:],1)
    np.fill_diagonal(C[:,1:],1)
    C[0,0] = 2
    C[n-1,n-1] = 7
    C[n-1,n-2] = 2

    P = np.array([points[0] + 2*points[1] if i == 0 
                  else 8*points[n-1]+points[n] if i == n-1
                  else 2*(2*points[i] + points[i + 1]) for i in range(n)])
    Px = P[:,0].reshape(n,1)
    Py = P[:,1].reshape(n,1)

    Ax = linear.solve_by_gauss(np.append(C,Px,axis=1))
    Ay = linear.solve_by_gauss(np.append(C,Py,axis=1))
    A = np.vstack((Ax,Ay)).T
    #A = np.linalg.solve(C, P)

    B = [0] * n
    for i in range(n - 1):
        B[i] = 2 * points[i + 1] - A[i + 1]
    B[n - 1] = (A[n - 1] + points[n]) / 2

    return A, B

def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * \
        t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d

# return one cubic curve for each consecutive points
def get_bezier_cubic(points):
    A, B = bezier_interpolate(points)
    return [ get_cubic(points[i], A[i], B[i], points[i + 1])
             for i in range(len(points) - 1) ]

# evalute each cubic curve on the range [0, 1] sliced in n points
def evaluate_bezier(points, n):
    if len(points) < 3 : return
    curves = get_bezier_cubic(points)
    return np.array([fun(t) for fun in curves for t in np.linspace(0, 1, n)])

#points = []
#fig = plt.figure(figsize=(5,5))
#plt.xlim(0,10)
#plt.ylim(0,10)

'''def onclick(event):
    #points.append(np.array([event.xdata,event.ydata]))
    plt.clf(); plt.xlim(0,10); plt.ylim(0,10)
    for x,y in points: plt.plot(x,y,'bo')
    if len(points) > 2:
        path = evaluate_bezier(points,50)
        px, py = path[:,0], path[:,1]
        plt.plot(px, py, 'b-')
        A,B = bezier_interpolate(points)
        for i in range(len(A)):
            plt.plot([A[i][0],points[i][0]], [A[i][1],points[i][1]], 'b-', color = 'black')
            plt.plot([B[i][0],points[i+1][0]], [B[i][1],points[i+1][1]], 'b-', color = 'black')
            plt.plot(A[i][0], A[i][1], '.', color = 'black')
            plt.plot(B[i][0], B[i][1], '.', color = 'black')
    #fig.canvas.draw()'''

#fig.canvas.mpl_connect('button_press_event', onclick)

#plt.show()