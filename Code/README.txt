RBE 595: Advanced Robot Navigation - Project 3: Non-Linear Kalman Filter

Name: Miheer Diwan
Robotics Engineering
Worcester Polytechnic Institute

The Code folder contains three files:
1. 'ekf_functions.py' has the ekf function definition
2. 'ekf_script.py' executes the functions on the input
3. 'utility.py' has the helper functions


Copy the data folder in the parent folder and run the script file for execution.
The outputs will be populated in the 'Outputs' folder

Covariance Matrix: 
R = np.array([
    [6.66963921e-03, 5.38548767e-05, 1.63620113e-03, 1.67040857e-05, 4.10176940e-03, 7.35551995e-04],
    [5.38548767e-05, 4.48424965e-03, -1.31399330e-03, -3.58369361e-03, 8.33674635e-04, -6.79055429e-05],
    [1.63620113e-03, -1.31399330e-03, 8.81311935e-03, 1.41573427e-03, 2.83727856e-03, -1.02644087e-03],
    [1.67040857e-05, -3.58369361e-03, 1.41573427e-03, 4.11917161e-03, 3.35394858e-04, -4.47555569e-04],
    [4.10176940e-03, 8.33674635e-04, 2.83727856e-03, 3.35394858e-04, 6.38801706e-03, -1.14463037e-03],
    [7.35551995e-04, -6.79055429e-05, -1.02644087e-03, -4.47555569e-04, -1.14463037e-03, 1.10043447e-03]
])