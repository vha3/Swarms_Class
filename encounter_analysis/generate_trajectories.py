import sys, pickle
import numpy
import numpy.linalg
from scipy import integrate
import random
from numpy.linalg import pinv
from scipy import interpolate

densities = numpy.array([1.17, 9.49e-2, 4.07e-3, 3.31e-4, 1.68e-5, 5.08e-7,
                         1.80e-8, 3.26e-9, 1.18e-9, 5.51e-10, 2.91e-10, 1.66e-10,
                         9.91e-11, 6.16e-11, 3.94e-11, 2.58e-11, 1.72e-11, 1.16e-11,
                         7.99e-12, 5.55e-12, 3.89e-12, 2.75e-12, 1.96e-12, 1.4e-12,
                         1.01e-12, 7.3e-13])*(1000./1)**3.
altitudes = numpy.arange(0, 520000, 20000)/1000.
density = interpolate.interp1d(altitudes, densities, 'linear')

h = .035*1000                                              # spacecraft height, mm
w = .035*1000                                              # spacecraft width, mm
d = .002*1000                                              # spacecraft depth, mm
Az = numpy.array([[h*d, h*d, h*h]]).T                      # z-direction coil surface (m^2)
A = h*h                                                    # surface area (mm^2)
Cd = 2.2                                                   # drag coefficient
mass = .006                                                # mass of spacecraft, kg
omega_earth = numpy.array([[0., 0., 7.2921159e-5]]).T      # angular velocity of Earth, rad/sec

mu_e = 398600.4418   # Earth graviational parameter, (km^3)/(sec^2)
J2   = 1.7555e10     # J2, (km^5)/(sec^2)
R_e  = 6371.         # Radius of Earth, km

effectiveArea = numpy.random.uniform(0., (h/1000./1000.)**2.)

def crs(vector):
    first = vector[0][0]
    second = vector[1][0]
    third = vector[2][0]
    return numpy.array([[0., -third, second],
                        [third, 0., -first],
                        [-second, first, 0.]])

def getVs(position, velocity):
    return velocity - numpy.dot(crs(omega_earth), position)

def getDensity(position):
    try:
        return density(numpy.linalg.norm(position) - R_e)
    except:
        print 'Collision with Earth'

def getDragDecoupled(X):
    position = numpy.array([X[0:3]]).T
    velocity = numpy.array([X[3:6]]).T

    Vs = getVs(position, velocity)
    Aeff = effectiveArea
    rho = getDensity(position)

    return 0.5*Aeff*rho*numpy.dot(Vs.T, Vs)[0][0]/mass*(-1*Vs/numpy.linalg.norm(Vs))

def orbitalDerivatives(X, t):
    x, y, z, xdot, ydot, zdot, dv = X

    R = (x**2. + y**2. + z**2.)**(1./2.)
    term0 = -mu_e/(R**3.)
    term1 = (5.*J2*(-x**2. - y**2. + 2.*(z**2.)))/(2.*(R**7.))
    term2 = J2/(R**5.)

    drag = getDragDecoupled(X)
    dvdot = numpy.linalg.norm(drag)

    xddot = term0*x + term1*x + term2*x + drag[0][0]
    yddot = term0*y + term1*y + term2*y + drag[1][0]
    zddot = term0*z + term1*z - term2*z*2. + drag[2][0]

    return [xdot, ydot, zdot, xddot, yddot, zddot, dvdot]

# Sample every 60 seconds
totaltime = numpy.arange(0, 353000, 60)
initial_position = numpy.array([374., 0., 0.])
initial_velocity = numpy.array([0., 4.76001, 6.01425])
initial_dv = numpy.array([0])

def runOrbitalSim(initial_position, initial_velocity):
    pos_init = ((initial_position/numpy.linalg.norm(initial_position))*
                (R_e+numpy.linalg.norm(initial_position)))
    X = numpy.hstack((pos_init, initial_velocity, initial_dv))
    return integrate.odeint(orbitalDerivatives, X, totaltime)

def runMonteCarlo(N, random=True):
    results = [0]*N
    for i in range(N):
        if random:
            vel = numpy.random.randn(3,1)
            vel = (vel/numpy.linalg.norm(vel))*numpy.random.uniform(0, .002)
        else:
            vel = numpy.array([numpy.random.randn(), 0., 0.])
            vel = (vel/numpy.linalg.norm(vel))*numpy.random.uniform(0, .002)
        effectiveArea = numpy.random.uniform(0., (h/1000./1000.)**2.)
        results[i] = runOrbitalSim(initial_position, initial_velocity+vel.T[0])
    return results

trajectories = runMonteCarlo(int(sys.argv[1]))

with open(sys.argv[2], 'w') as ifile:
  pickle.dump(trajectories, ifile)

# for t in trajectories:
#   rows, cols = t.shape
#   for r in range(rows):
#     print ','.join(map(str, t[r]))

sys.exit()
