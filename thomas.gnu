set term pngcairo size 49cm, 27cm

# general settings
set grid
set xlabel 'x'; set ylabel 'y'; set zlabel 'z'; set ticslevel 0
set parametric
set output 'plot.png'

# define rhs of dynamical system
b = 0.2
rhsx(t, x, y, z) = sin(y) - b*x
rhsy(t, x, y, z) = sin(z) - b*y
rhsz(t, x, y, z) = sin(x) - b*z

# time step-length
dt = 1.0e-3

# initial condition/current values
x0 = 1; y0 = 0; z0 = 0
x1 = x0; y1 = y0; z1 = z0

# define helper
k1x = 0; k1y = 0; k1z = 0
k2x = 0; k2y = 0; k2z = 0
k3x = 0; k3y = 0; k3z = 0
k4x = 0; k4y = 0; k4z = 0

# integration duration
tsim = 1000
set trange [0:tsim]
set samples floor(tsim/dt)

spl '+' u (\
  x0 = x1, \
  y0 = y1, \
  z0 = z1, \
  k1x = rhsx($1, x0, y0, z0), \
  k1y = rhsy($1, x0, y0, z0), \
  k1z = rhsz($1, x0, y0, z0), \
  k2x = rhsx($1 + 0.5*dt, x0 + 0.5*dt*k1x, y0 + 0.5*dt*k1y, z0 + 0.5*dt*k1z), \
  k2y = rhsy($1 + 0.5*dt, x0 + 0.5*dt*k1x, y0 + 0.5*dt*k1y, z0 + 0.5*dt*k1z), \
  k2z = rhsz($1 + 0.5*dt, x0 + 0.5*dt*k1x, y0 + 0.5*dt*k1y, z0 + 0.5*dt*k1z), \
  k3x = rhsx($1 + 0.5*dt, x0 + 0.5*dt*k2x, y0 + 0.5*dt*k2y, z0 + 0.5*dt*k2z), \
  k3y = rhsy($1 + 0.5*dt, x0 + 0.5*dt*k2x, y0 + 0.5*dt*k2y, z0 + 0.5*dt*k2z), \
  k3z = rhsz($1 + 0.5*dt, x0 + 0.5*dt*k2x, y0 + 0.5*dt*k2y, z0 + 0.5*dt*k2z), \
  k4x = rhsx($1 + dt, x0 + dt*k3x, y0 + dt*k3y, z0 + dt*k3z), \
  k4y = rhsy($1 + dt, x0 + dt*k3x, y0 + dt*k3y, z0 + dt*k3z), \
  k4z = rhsz($1 + dt, x0 + dt*k3x, y0 + dt*k3y, z0 + dt*k3z), \
  x1 = x0 + dt/6.0*(k1x + 2*k2x + 2*k3x + k4x), \
  y1 = y0 + dt/6.0*(k1y + 2*k2y + 2*k3y + k4y), \
  z1 = z0 + dt/6.0*(k1z + 2*k2z + 2*k3z + k4z), \
  x1):(y1):(z1) w lines notitle

unset output