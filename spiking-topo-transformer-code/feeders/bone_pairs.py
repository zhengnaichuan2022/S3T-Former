# NTU RGB+D 25-joint bone vectors (1-based indices): child -> parent direction
ntu_pairs = (
    (2, 3), (2, 5), (3, 4), (5, 6), (6, 7), (2, 8), (8, 9), (8, 11),
    (9, 10), (5, 11), (11, 12), (12, 13), (1, 14), (1, 15), (1, 1),
    (14, 16), (15, 17),
)

# NW-UCLA / Kinect 20-joint topology (1-based), same convention as common UCLA feeders
ucla_pairs = (
    (1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3),
    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19),
)
