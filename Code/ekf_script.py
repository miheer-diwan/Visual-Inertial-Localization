import numpy as np
from utils import plotter,estimate_covariances,plotter_2
from ekf_functions import EKF

filenames = [
        'studentdata0.mat',
        'studentdata1.mat',
        'studentdata2.mat',
        'studentdata3.mat',
        'studentdata4.mat',
        'studentdata5.mat',
        'studentdata6.mat',
        'studentdata7.mat'
]

# Task 2:
for filename in filenames:
        # print()
        print(f"\n Plotting for : {filename}")
        # print()
        plotter(filename)


## Task 3:
R = np.zeros((6, 6))

for filename in filenames:
     R_estimate = estimate_covariances(filename)
    #    print('pucca')
     R += R_estimate

# Calculate the average R
R = R / len(filenames)

print("\n R: \n")
print(R)

## Task 4:

# Initialize covariance matrix P
P = np.eye(15)*0.01

# Define process noise covariance Q
Q = np.eye(15) * 1e-6

R = np.array([
        [6.66963921e-03, 5.38548767e-05, 1.63620113e-03, 1.67040857e-05, 4.10176940e-03, 7.35551995e-04],
        [5.38548767e-05, 4.48424965e-03, -1.31399330e-03, -3.58369361e-03, 8.33674635e-04, -6.79055429e-05],
        [1.63620113e-03, -1.31399330e-03, 8.81311935e-03, 1.41573427e-03, 2.83727856e-03, -1.02644087e-03],
        [1.67040857e-05, -3.58369361e-03, 1.41573427e-03, 4.11917161e-03, 3.35394858e-04, -4.47555569e-04],
        [4.10176940e-03, 8.33674635e-04, 2.83727856e-03, 3.35394858e-04, 6.38801706e-03, -1.14463037e-03],
        [7.35551995e-04, -6.79055429e-05, -1.02644087e-03, -4.47555569e-04, -1.14463037e-03, 1.10043447e-03]
])

for filename in filenames:
    print(f"\nFiltering {filename}")

    ekf = EKF(filename, Q, R)

    filtered_positions, filtered_orientations = ekf.iterator(filename)

    print()
    print(f"Plotting: {filename}")
    print()

    plotter_2(filename , filtered_positions)