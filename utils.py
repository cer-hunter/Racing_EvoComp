import math
# Corner definitions

# Angle (radians) of straights before and after corner
STRAIGHT_ANGLES = [-1.5137184858322144, 1.6283349990844727]

# x and y coordinate of corner entry
CORNER_ENTRY_COORD = [230.2320556640625, -4.395766735076904]


def to_corner_coord(x, y):
    entry_x, entry_y = CORNER_ENTRY_COORD
    angle = STRAIGHT_ANGLES[0]

    # Calculate the relative position of the coordinate with respect to the corner entry
    relative_x = x - entry_x
    relative_y = y - entry_y

    # Rotate the relative position based on the angle
    cx = relative_x * math.cos(angle) + relative_y * math.sin(angle)
    cy = -relative_x * math.sin(angle) + relative_y * math.cos(angle)

    return cx, cy


def to_world_coord(cx, cy):
    entry_x, entry_y = CORNER_ENTRY_COORD
    angle = STRAIGHT_ANGLES[0]

    # Rotate the relative position based on the negative angle
    relative_x = cx * math.cos(-angle) + cy * math.sin(-angle)
    relative_y = -cx * math.sin(-angle) + cy * math.cos(-angle)

    # Calculate the world position of the coordinate
    x = relative_x + entry_x
    y = relative_y + entry_y

    return x, y

def to_entry_angle(angle):
    return angle - STRAIGHT_ANGLES[0]

def to_exit_angle(angle):
    return angle - STRAIGHT_ANGLES[1]