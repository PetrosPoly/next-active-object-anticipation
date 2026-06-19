# import rerun as rr
# import numpy as np

# rr.init("rerun_example_clear", spawn=True)

# vectors = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]  # Example vectors
# origins = [np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])]  # Example origins
# colors = [np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])]  # Example colors

# for i, (vector, origin, color) in enumerate(zip(vectors, origins, colors)): 
#     rr.log(f"arrows/{i}", rr.Arrows3D(vectors=vector, origins=origin, colors=color))

# for i in range(len(vectors)):
#     rr.log(f"arrows/{i}", rr.Clear(recursive=False))  # or `rr.Clear.flat()`

import rerun as rr

rr.init("rerun_example_line_strip3d_batch", spawn=True)

rr.log(
    "strips",
    rr.LineStrips3D(
        [
            [
                [0, 0, 2],
                [1, 0, 2],
                [1, 1, 2],
                [0, 1, 2],
            ],
            [
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 1],
            ],
        ],
        colors=[[255, 0, 0], [0, 255, 0]],
        radii=[0.025, 0.005],
        labels=["one strip here", "and one strip there"],
    ),
)