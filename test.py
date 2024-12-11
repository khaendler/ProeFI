import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual data)
data = [
    {1: 4},  # Creation at time 1, elimination at time 4
    {3: 8},
    {5: 10},
    {7: 12},
    {9: 15}
]

# Example stats (values spanning the same time range as the events)
# Replace with your actual stats
stats_times = np.linspace(1, 15, 100)  # Times from start to finish
stats_values = np.sin(stats_times) * 50 + 100  # Example stats values

# Extract events
events = []
for entry in data:
    for creation, elimination in entry.items():
        events.append((creation, 'create'))
        events.append((elimination, 'eliminate'))

# Sort events by time
events.sort(key=lambda x: x[0])

# Calculate active instances
active_instances = 0
times = []
active_count = []

for event in events:
    times.append(event[0])
    if event[1] == 'create':
        active_instances += 1
    elif event[1] == 'eliminate':
        active_instances -= 1
    active_count.append(active_instances)

# Normalize stats values to match the range of active instance counts
stats_values_normalized = (stats_values - stats_values.min()) / (stats_values.max() - stats_values.min()) * max(active_count)

# Plot the data
plt.figure(figsize=(10, 6))

# Plot the stats in the background
plt.fill_between(stats_times, stats_values_normalized, color='lightblue', alpha=0.5, label='Background Stats')

# Plot the active instances
plt.step(times, active_count, where='post', label='Active Instances', color='red')

# Add labels, legend, and title
plt.xlabel('Time')
plt.ylabel('Count / Normalized Stats')
plt.title('Active Instances with Background Stats')
plt.legend()
plt.grid(True)
plt.show()