# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self, N_=4, dt_= 0.1, q_=0.1):
        self.N  = N_  # process model dimension
        self.dt = dt_ # time increment
        self.q  = q_  # process noise variable for Kalman filter Q

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############

        dt = self.dt
        #
        if (self.N ==4):
            return np.matrix([
                [1, 0, dt, 0 ],
                [0, 1, 0,  dt],
                [0, 0, 1,  0 ],
                [0, 0, 0,  1 ],
            ])
        if (self.N ==6):
            return np.matrix([
                [1, 0, 0, dt, 0,  0 ],
                [0, 1, 0, 0,  dt, 0 ],
                [0, 0, 1, 0,  0,  dt],
                [0, 0, 0, 1,  0,  0 ],
                [0, 0, 0, 0,  1,  0 ],
                [0, 0, 0, 0,  0,  1 ],
            ])
        
        print("ERROR: N not supported")
        return 0
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############

        q = self.q
        dt = self.dt
        q1 = ((dt**3)/3) * q 
        q2 = ((dt**2)/2) * q 
        q3 = dt * q

        if (self.N ==4):
            return np.matrix([
                [q1, 0,  q2, 0 ],
                [0,  q1, 0,  q2],
                [q2, 0,  q3, 0 ],
                [0,  q2, 0,  q3],
            ])
        if (self.N ==6):
            return np.matrix([
                [q1, 0,  0,  q2, 0,  0 ],
                [0,  q1, 0,  0,  q2, 0 ],
                [0,  0,  q1, 0,  0,  q2],
                [q2, 0,  0,  q3, 0,  0 ],
                [0,  q2, 0,  0,  q3, 0 ],
                [0,  0,  q2, 0,  0,  q3],
            ])

        print("ERROR: N not supported")
        return 0
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        F = self.F()
        x = track.x
        P = track.P
        #
        x = F*x # state prediction
        P = F*P*F.transpose() + self.Q() # covariance prediction
        #
        track.set_x(x)
        track.set_P(P)
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        
        x = track.x
        P = track.P
        H = meas.sensor.get_H(x) # measurement matrix
        #
        gamma = self.gamma(track, meas)
        S = self.S(track, meas, H)
        #
        K = P*H.transpose()*np.linalg.inv(S) # Kalman gain
        I = np.identity(self.N)
        #
        x = x + K*gamma # state update
        P = (I - K*H) * P # covariance update
        #
        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############

        x = track.x
        z = meas.z
        hx = meas.sensor.get_hx(x)
        #
        gamma = z - hx # residual
        #
        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############

        P = track.P
        R = meas.R
        #
        S = H*P*H.transpose() + R # covariance of residual
        #
        return S
        
        ############
        # END student code
        ############ 

