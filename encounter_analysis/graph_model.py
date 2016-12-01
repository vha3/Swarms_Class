import sys, pickle, os

def distance(s0, s1):
  x0, y0, z0 = s0
  x1, y1, z1 = s1
  return ((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)**0.5

def within_threshold(s0, s1, threshold):
  x0, y0, z0 = s0
  x1, y1, z1 = s1

  x_d = x1 - x0
  y_d = y1 - y0
  z_d = z1 - z0

  if abs(x_d) > threshold or abs(y_d) > threshold or abs(z_d) > threshold:
    return False
  else:
    return (x_d**2 + y_d**2 + z_d**2)**0.5 < threshold

# Understand how to vectorize
# Transform the data into a better format (no lists)
# Refactor this --> iteration is what is slowing it

# This should be renamed to traj count or something

if __name__ == '__main__':
  trajectories_path = sys.argv[1]

  for f in os.listdir(trajectories_path):
    if '.' not in f:
      print f
      trajectories_file = os.path.join(trajectories_path, f)
      with open(trajectories_file, 'r') as ifile:
        trajectories = pickle.load(ifile)

      # Get the dimensions
      num_sprites = len(trajectories)
      time_steps, values = trajectories[0].shape

      distance_threshold = 0.2 # in km

      num_meetings = [0] * time_steps
      for t in range(time_steps):
        # print 'Timestep: %d' % t
        for s0 in range(num_sprites):
          sprite0 = trajectories[s0][t][:3]
          for s1 in range(s0+1, num_sprites):
            sprite1 = trajectories[s1][t][:3]
            if within_threshold(sprite0, sprite1, distance_threshold):
              num_meetings[t] += 1

      with open(os.path.join(sys.argv[2], f+'_meetings'), 'w') as ofile:
        for m in num_meetings:
          ofile.write(str(m)+'\n')
