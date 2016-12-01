import sys, subprocess, time, os

# Set this up to maintain 5 running processing until 1000 runs are generated.
max_running = int(sys.argv[1])
num_runs = int(sys.argv[2])
output_dir = sys.argv[3]

process_count = 0
slots = range(max_running)
processes = {}

while process_count < num_runs:
  # Check for completed processes and append open slots to slots
  for s in processes:
    if processes[s].poll() is not None:
      slots.append(s)

  # Replace open slots with running processes
  while slots and process_count < num_runs:
    processes[slots.pop()] = subprocess.Popen(['python', 'generate_trajectories.py', '100', os.path.join(output_dir, str(process_count))])
    process_count += 1
    print 'Process count is now %d' % process_count

  # Sleep for 1 minute
  time.sleep(60)

sys.exit()
