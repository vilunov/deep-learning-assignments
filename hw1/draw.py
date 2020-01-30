import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(history, points):
    key_frames_mult = len(history) // 500
    fig, ax = plt.subplots()
    (line,) = plt.plot([], [], lw=2)

    def init():
        x = list(points[history[0], 0])
        y = list(points[history[0], 1])
        plt.plot(x, y, "co", label="oi")

        extra_x = (max(x) - min(x)) * 0.05
        extra_y = (max(y) - min(y)) * 0.05
        ax.set_xlim(max(x) + extra_x, min(x) - extra_x)
        ax.set_ylim(max(y) + extra_y, min(y) - extra_y)

        line.set_data([], [])
        return (line,)

    def update(frame):
        x = list(points[history[frame], 0]) + [points[history[frame][0], 0]]
        y = list(points[history[frame], 1]) + [points[history[frame][0], 1]]
        line.set_data(x, y)
        return line

    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, len(history), key_frames_mult),
        init_func=init,
        interval=1,
        repeat=False,
    )
    ani.save(filename="solution.mp4")
    plt.show()
