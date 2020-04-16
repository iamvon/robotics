import numpy as np
import matplotlib.pyplot as plt
import math


dt = 0.1
ALLTIME = 50

R = np.diag([
    0.1,
    0.1,
    np.deg2rad(1.0),
    1.0,
    1.0
])**2#nxn, covariance of noise in motion translation 
G = None#nxn, jacobian of motion translation
H = None#mxn, jacobian of measurement translation
Q = np.diag([
    1.0,
    1.0
])**2 #mxm, covariance of noise in measurement translation, The larger value of Q is, the less K gain'value is and the less state of robot depends on measurement value. We can use EM algorithm like in pykalman library to compute Q and R's value
illustation = True

def jacobian_G(x, u): # {x,y,yaw,v, a}, {a, omega}
    
    yaw = x[2, 0]
    v = x[3, 0] + u[1, 0]*dt

    R = np.array(
        [[1, 0, -v*np.sin(yaw)*dt , np.cos(yaw)*dt, 1/2*dt*dt],
        [0, 1, v*np.cos(yaw)*dt, np.sin(yaw)*dt, 1/2*dt*dt],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0]
        ])

    return R

def jacobian_H():

    H = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0]
    ])

    return H

# return input vector [a, w]
def calc_input():
    
    a = 0.5 #m/s2
    w = 0.1 #rad/s
    return np.array([[a], [w]])

def motion_model(x_prev, u_t):
    theta = x_prev[2, 0]
    F = np.array([
        [1, 0, 0, math.cos(theta)*dt, 0],
        [0, 1, 0, math.sin(theta)*dt, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)
    B = np.array([
        [1/2*dt*dt, 0],
        [1/2*dt*dt, 0],
        [0, dt],
        [dt, 0],
        [0, 0]
    ], dtype=np.float32)
    x = F @ x_prev + B @ u_t
    #print("Test : ", x)
    return x

def measurement_model(x):
    
    C = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ])
    z = C @ x
    return z

def observation(x_true):
    C = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ])
    tmp = np.diag([0.5, 0.5]) ** 2 
    z = C @ x_true + tmp @ np.random.randn(2,1)
    return z


def ekf_filter(x_prev, cov_prev, u_t, z_t):
    
    H = jacobian_H()

    x_pred = motion_model(x_prev, u_t)
    G = jacobian_G(x_pred, u_t)
    cov_pred = G@cov_prev@G.T + R #G(nxn) is jacobian of motion_model, R is covariance of noise in motion_model(n state, so R:nxn)
    K = cov_pred @ H.T @ np.linalg.inv(H@cov_pred@H.T + Q) # H is jacobian of measurement_model, Q is covariance of noise in measurement model. nxm (m is size of measurement)    
    delta_y = z_t - measurement_model(x_pred)
    x_est = x_pred + K @ delta_y #n
    cov_est = (np.eye(5) - K@H)@cov_pred
    print(cov_est.shape)
    print(cov_pred)
    
    return x_est, cov_est


def main():
    
    print("Start running EKF filter")
    
    x_true = np.zeros((5, 1))
    x_est = np.zeros((5, 1))
    cov_est = np.eye(5)

    x_rd = np.zeros((5, 1))

    t = 0

    #illustration
    hx_est = x_est
    hx_true = x_true
    hz = np.zeros((2,1))

    while t < ALLTIME:
        u = calc_input()
        x_true = motion_model(x_true, u)
        z = observation(x_true)
        x_est, cov_est = ekf_filter(x_est, cov_est, u, z)
        
        hx_est = np.hstack((hx_est, x_est))
        hx_true = np.hstack((hx_true, x_true))
        hz = np.hstack((hz,z))
        t = t + dt

        if illustation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hx_true[0, :].flatten(),
                     hx_true[1, :].flatten(), "-b")
            plt.plot(hx_est[0, :].flatten(),
                     hx_est[1, :].flatten(), "-r")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
    plt.pause(1000)

if __name__ == "__main__":
    main()


