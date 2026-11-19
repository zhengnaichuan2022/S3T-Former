# Bone vectors use 1-based (child, parent) pairs: bone[child] = joint[child] - joint[parent].

# NTU RGB+D 25-joint skeleton (the spine-shoulder joint 21 is the root and maps to itself).
ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12),
)

# NW-UCLA 20-joint skeleton (the shoulder-centre joint 3 is the root and maps to itself).
ucla_pairs = (
    (1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), (9, 3),
    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19),
)
