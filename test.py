import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()

# Example values
t_s_window = 100
cur_satellite = 2
t_e_window = 200
t_s = 50
t_e = 150

# Create and add rectangles
rect1 = patches.Rectangle((t_s_window, cur_satellite - 0.1), t_e_window - t_s_window, 0.2, edgecolor='k', fill=True)
ax.add_patch(rect1)

rect2 = patches.Rectangle((t_s, cur_satellite - 0.25), t_e - t_s, 0.5, edgecolor='k', fill=True)
ax.add_patch(rect2)

# Set axis limits
ax.set_xlim(0, 250)  # Adjust the limits as needed
ax.set_ylim(0, 5)

# Show the plot
plt.show()
