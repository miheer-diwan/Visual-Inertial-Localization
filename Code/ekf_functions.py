import sympy as sp
import numpy as np
from utils import estimate_pose
import scipy.io

class EKF:
    def __init__(self, filename, Q, R):
        self.file_path = 'data\\' + filename
        self.filter_pos = []
        self.filter_ori = []
        self.Q = Q
        self.R = R

    def predict_step(self, x, P, u, del_t):
        """
        Predict the state and covariance forward in time.
        :param u: control input (IMU readings)
        :param del_t: time step
        """
    # Update state with process model
        F, x_hat = self.process_model_step(x, del_t, u)

        # Predict covariance
        P = F @ P @ F.T + self.Q

        return x_hat, P

    def update_step(self, x_hat, data, P):
        """
        Update the state estimate using observed measurements.
        :param z: measurement vector
        """
        # Compute measurement matrix H
        H = self.Measurement_Jacobian()

        P = P.astype(np.float64)
        H = H.astype(np.float64)
        R = self.R.astype(np.float64)

        # Kalman gain
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

        # Update state estimate
        pos, ori, ts = estimate_pose(data)

        z = np.concatenate((pos.reshape(-1, 1), ori.reshape(-1, 1)))
        
        y = z - H @ x_hat

        x_hat += K @ y

        P = (np.eye(15) - K @ H) @ P

        return x_hat, P

    def process_model_step(self, x, dtime, u):
        """
        this fn. implements the process model
        """
        p1 = sp.symbols('p1')
        p2 = sp.symbols('p2')
        p3 = sp.symbols('p3')
        phi = sp.symbols('phi')
        theta = sp.symbols('theta')
        psi = sp.symbols('psi')
        p1_dot = sp.symbols('p1_dot')
        p2_dot = sp.symbols('p2_dot')
        p3_dot = sp.symbols('p3_dot')
        bg1 = sp.symbols('bg1')
        bg2 = sp.symbols('bg2')
        bg3 = sp.symbols('bg3')
        ba1 = sp.symbols('ba1')
        ba2 = sp.symbols('ba2')
        ba3 = sp.symbols('ba3')
        del_t = sp.symbols('del_t')

        wx = sp.symbols('wx')
        wy = sp.symbols('wy')
        wz = sp.symbols('wz')
        vx = sp.symbols('vx')
        vy = sp.symbols('vy')
        vz = sp.symbols('vz')


        F_sym, x_hat_sym = self.Symbolic_Jacobian()

        # Substituting 
        F = F_sym.subs(
            {p1: x[0, 0], 
             p2: x[1, 0], 
             p3: x[2, 0], 
             phi: x[3, 0], 
             theta: x[4, 0], 
             psi: x[5, 0], 
             p1_dot: x[6, 0],
             p2_dot: x[7, 0], 
             p3_dot: x[8, 0], 
             bg1: x[9, 0], 
             bg2: x[10, 0], 
             bg3: x[11, 0], 
             ba1: x[12, 0],
             ba2: x[13, 0], 
             ba3: x[14, 0], 
             wx: u[0, 0], 
             wy: u[1, 0], 
             wz: u[2, 0], 
             vx: u[3, 0], 
             vy: u[4, 0],
             vz: u[5, 0], 
             del_t: dtime})
        
        x_hat = x_hat_sym.subs(
            {p1: x[0, 0], 
             p2: x[1, 0], 
             p3: x[2, 0], 
             phi: x[3, 0], 
             theta: x[4, 0], 
             psi: x[5, 0], 
             p1_dot: x[6, 0],
             p2_dot: x[7, 0], 
             p3_dot: x[8, 0], 
             bg1: x[9, 0], 
             bg2: x[10, 0], 
             bg3: x[11, 0], 
             ba1: x[12, 0],
             ba2: x[13, 0], 
             ba3: x[14, 0], 
             wx: u[0, 0], 
             wy: u[1, 0], 
             wz: u[2, 0], 
             vx: u[3, 0], 
             vy: u[4, 0],
             vz: u[5, 0], 
             del_t: dtime})
        
        F = np.array(F).reshape(15, 15)

        x_hat = np.array(x_hat).reshape(15, 1)

        return F, x_hat

    def Symbolic_Jacobian(self):
 
        p1, p2, p3, phi, theta, psi, p1_dot, p2_dot, p3_dot, bg1, bg2, bg3, ba1, ba2, ba3, del_t = sp.symbols(
            'p1 p2 p3 phi theta psi p1_dot p2_dot p3_dot bg1 bg2 bg3 ba1 ba2 ba3 del_t')

        # Define the matrix elements
        G_q = sp.Matrix([
            [sp.cos(theta), 0, -sp.cos(phi) * sp.sin(theta)],
            [0, 1, sp.sin(phi)],
            [sp.sin(theta), 0, sp.cos(phi) * sp.cos(theta)]
        ])

        # Compute the inverse of G_q
        G_q_inv = G_q.inv()

        # Write R_q as a 3x3 matrix just like G_q
        R_q = sp.Matrix([
            [sp.cos(psi) * sp.cos(theta) - sp.sin(phi) * sp.sin(theta) * sp.sin(psi),
             -sp.cos(phi) * sp.sin(psi),
             sp.cos(psi) * sp.sin(theta) + sp.cos(theta) * sp.sin(phi) * sp.sin(psi)],

            [sp.cos(psi) * sp.sin(phi) * sp.sin(theta) + sp.cos(theta) * sp.sin(psi),
             sp.cos(phi) * sp.cos(psi),
             sp.sin(psi) * sp.sin(theta) - sp.cos(psi) * sp.cos(theta) * sp.sin(phi)],

            [-sp.cos(phi) * sp.sin(theta), sp.sin(phi), sp.cos(phi) * sp.cos(theta)]
        ])

        # vector in state space
        x = sp.Matrix([p1, p2, p3, phi, theta, psi, p1_dot, p2_dot, p3_dot, bg1, bg2, bg3, ba1, ba2, ba3])

        p_dot = sp.Matrix([p1_dot, p2_dot, p3_dot])

        # Define inputs
        wx, wy, wz, vx, vy, vz = sp.symbols('wx wy wz vx vy vz')
        u_w = sp.Matrix([wx, wy, wz])
        u_a = sp.Matrix([vx, vy, vz])
        g = sp.Matrix([0, 0, -9.81])
        n_bg = sp.Matrix([0, 0, 0])
        n_ba = sp.Matrix([0, 0, 0])

        # state is given by
        x_dot = sp.Matrix([p_dot, G_q_inv * u_w, g + R_q * u_a, n_bg, n_ba])

        # Get A from x_dot jacobian w.r.t x
        A = x_dot.jacobian(x)

        F = sp.eye(15) + A * del_t

        x_hat = x + x_dot * del_t

        return F, x_hat

    def Measurement_Jacobian(self):
        """
        this fn. computes the Jacobian for the measurement model
        """
        H = np.zeros((6, 15))

        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        return H

    def iterator(self, filename):

        if(filename =='studentdata0.mat'):
            student_number = 0
        else:
            student_number = 1
           
        i = 0
        student_data = scipy.io.loadmat(self.file_path, simplify_cells=True)
        # Extract motion capture data
        vicon = student_data['vicon']
        vicon_time = student_data['time']
        vicon_data = np.array(vicon).T

        # Extract ground truth positions and orientations
        gt_pos = vicon_data[0, :3]
        gt_ori = vicon_data[0, 3:6]
        gt_vel = vicon_data[0, 6:9]

        x_hat = np.hstack((gt_pos, gt_ori, gt_vel, [0, 0, 0],
                                                [0, 0, 0])).reshape(15, 1)

        #Defining the P matrix as diagonal matrix with covariance value as 0.00
        P = np.eye(15) * 0.01

        print("Filtering", self.file_path)

        for data in student_data['data']:
            # ENd condition for the for loop
            if i == len(student_data['data'])-1:
                break
            # Calculating the del_t
            del_t = student_data['data'][i + 1]['t'] - student_data['data'][i]['t']

            # The data for the student 0 is different than others contering that here
            if student_number==0:
                omg = np.array(data['drpy']).reshape(-1, 1)

            else :
                omg = np.array(data['omg']).reshape(-1, 1)

            acc = np.array(data['acc']).reshape(-1, 1)

            # Creating a u verctor(6,1)
            u = np.concatenate((omg, acc))

            tag_ids = data['id']

            if isinstance(tag_ids, int):
                tag_ids = np.array([tag_ids])

            if len(tag_ids) == 0:
                x_hat, P = self.predict_step(x_hat, P, u, del_t)

            else:
                x_hat, P = self.predict_step(x_hat, P, u, del_t)
                x_hat, P = self.update_step(x_hat, data, P)

            self.filter_pos.append([x_hat[0, 0], x_hat[1, 0], x_hat[2, 0]])
            self.filter_ori.append([x_hat[3, 0], x_hat[4, 0], x_hat[5, 0]])

            i += 1

        return self.filter_pos, self.filter_ori