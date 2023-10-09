from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string
import time

PRINT_GRID = True
PRINT_LOCATIONS = False

SCREEN_HEIGHT = 240
SCREEN_WIDTH = 256
MATCH_THRESHOLD = 0.9

MASK_COLOUR = np.array([252, 136, 104])

image_files = {
    "mario": {
        "small": [
            "Images/marioA.png",
            "Images/marioB.png",
            "Images/marioC.png",
            "Images/marioD.png",
            "Images/marioE.png",
            "Images/marioF.png",
            "Images/marioG.png",
        ],
        "tall": [
            "Images/tall_marioA.png",
            "Images/tall_marioB.png",
            "Images/tall_marioC.png",
        ],
    },
    "enemy": {
        "goomba": ["Images/goomba.png"],
        "koopa": ["Images/koopaA.png", "Images/koopaB.png"],
    },
    "pipe": {
        "pipe": ["Images/pipe_upper_section.png", "Images/pipe_lower_section.png"],
    },
    "block": {
        "block": ["Images/block1.png", "Images/block2.png", "Images/block3.png", "Images/block4.png"],
        "question_block": ["Images/questionA.png", "Images/questionB.png", "Images/questionC.png"],
    },
    "hole": {
        "hole": ["Images/hole.jpeg", "Images/hole2.jpeg"],
    },
    "item": {
        "mushroom": ["Images/mushroom_red.png"],
    },
}

def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0] * image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None
    dimensions = tuple(template.shape[::-1])
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

include_flipped = {"mario", "enemy"}

templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

colour_map = {
    (104, 136, 252): " ",  # sky blue colour
    (0, 0, 0): " ",  # black
    (252, 252, 252): "'",  # white / cloud colour
    (248, 56, 0): "M",  # red / mario colour
    (228, 92, 16): "%",  # brown enemy / block colour
}
unused_letters = sorted(set(string.ascii_uppercase) - set(colour_map.values()), reverse=True)
DEFAULT_LETTER = "?"

def _get_colour(colour):
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]

    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER

def print_grid(obs, object_locations):
    pixels = {}

    for category in object_locations:
        for location, dimensions, object_name in object_locations[category]:
            x, y = location
            width, height = dimensions
            name_str = object_name.replace("_", "-") + "-"
            for i in range(width):
                pixels[(x + i, y)] = name_str[i % len(name_str)]
                pixels[(x + i, y + height - 1)] = name_str[(i + height - 1) % len(name_str)]
            for i in range(1, height - 1):
                pixels[(x, y + i)] = name_str[i % len(name_str)]
                pixels[(x + width - 1, y + i)] = name_str[(i + width - 1) % len(name_str)]

    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                colour = pixels[coords]
            else:
                colour = _get_colour(obs[y][x])
            line.append(colour)

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        if stop_early and locations:
            break

    return [(loc, locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["pipe"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["pipe"]["pipe"][1]

    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))

    if not upper_locs:
        return []

    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y + h, x + 2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations

def locate_objects(screen, mario_status):
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            if object_name == "pipe":
                continue

            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    object_locations["pipe"] += _locate_pipe(screen)

    return object_locations

def find_mario_location(screen, info, step, env, prev_action):
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    mario_locations = object_locations["mario"]
    mario_position = None
    if len(mario_locations) > 0:
        mario_position = mario_locations[0][0]

    return mario_position

def detect_objects(screen, object_image):
    obs = cv.cvtColor(screen, cv.COLOR_BGR2RGB)
    cv.imwrite("Images/screen.jpeg", obs)
    obs = cv.imread("Images/screen.jpeg", cv.IMREAD_COLOR)
    hole = cv.imread(object_image, cv.IMREAD_COLOR)

    result = cv.matchTemplate(obs, hole, cv.TM_CCOEFF_NORMED)

    w = hole.shape[1]
    h = hole.shape[0]

    yloc, xloc = np.where(result >= 0.8)
    rectangles = []

    for (x, y) in zip(xloc, yloc):
        rectangles.append([int(x), int(y), int(w), int(h)])
        rectangles.append([int(x), int(y), int(w), int(h)])

    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.2)

    return rectangles

def get_object_locations(screen, object_image):
    object_locations = detect_objects(screen, object_image)
    object_position = None

    if len(object_locations) > 0:
        object_position_x = object_locations[-1][0]
        object_position_y = object_locations[-1][1]
        object_position = (object_position_x, object_position_y)

    return object_position

def find_object_location(screen, info, step, env, prev_action):
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    mario_locations = object_locations["mario"]
    mario_position = None
    if len(mario_locations) > 0:
        mario_position = mario_locations[0][0]

    enemy_locations = object_locations["enemy"]
    enemy_position = None
    if len(enemy_locations) > 0:
        enemy_position = enemy_locations[0][0]

    pipe_locations = object_locations["pipe"]
    pipe_position = None
    if len(pipe_locations) > 0:
        pipe_position = pipe_locations[0][0]

    hole_position = get_object_locations(screen, "Images/hole2.jpeg")
    first_left_stair = get_object_locations(screen, "Images/left_stair1.jpeg")
    first_right_stair = get_object_locations(screen, "Images/right_stair1.jpeg")
    second_stair_position = get_object_locations(screen, "Images/stair3.jpeg")
    last_stair_position = get_object_locations(screen, "Images/last.jpeg")
    goal_position = get_object_locations(screen, "Images/goal.jpeg")

    if last_stair_position is None:
        last_stair_position = get_object_locations(screen, "Images/last2.jpeg")

        if last_stair_position is None:
            last_stair_position = get_object_locations(screen, "Images/last3.jpeg")

    next_object = nearest_object(mario_position, enemy_position, pipe_position, hole_position, first_left_stair, first_right_stair, second_stair_position, last_stair_position, goal_position)

    if enemy_position == next_object:
        return ["enemy", enemy_position]
    elif pipe_position == next_object:
        return ["pipe", pipe_position]
    elif hole_position == next_object:
        return ["hole", hole_position]
    elif first_left_stair == next_object:
        return ["first left stair", first_left_stair]
    elif first_right_stair == next_object:
        return ["first right stair", first_right_stair]
    elif second_stair_position == next_object:
        return ["second stair", second_stair_position]
    elif last_stair_position == next_object:
        return ["last stair", last_stair_position]
    elif goal_position == next_object:
        return ["goal", goal_position]
    else:
        return None


def nearest_object(mario_position, enemy_position, pipe_position, hole_position, first_left_stair, first_right_stair, second_stair_location, last_stair_location, goal_position):
    objects = []
    locations = []

    for object in enemy_position, pipe_position, hole_position, first_left_stair, first_right_stair, second_stair_location, last_stair_location, goal_position:
        if object is not None:
            objects.append(object)
            location = object[0] - mario_position[0]
            locations.append(location)

    if len(objects) == 0:
        return (0, 0)

    return objects[locations.index(find_min_location(locations))]


def find_min_location(locations):
    min_location = max(locations)

    for location in locations:
        if location > 0:
            if location < min_location:
                min_location = location

    return min_location

def make_action(screen, info, step, env, prev_action):
    mario = find_mario_location(screen, info, step, env, prev_action)
    object = find_object_location(screen, info, step, env, prev_action)

    action = 3

    last_stair = False

    if object is None:
        action = 3
    else:
        if object[0].lower() == "pipe":
            action = jump_pipe(mario, object[1])
        elif object[0].lower() == "enemy":
            action = jump_enemy(mario, object[1])
        elif object[0].lower() == "hole":
            action = jump_hole(mario, object[1])
        elif object[0].lower() == "first left stair":
            action = jump_first_left_stair(mario, object[1])
        elif object[0].lower() == "first right stair":
            action = jump_first_right_stair(mario, object[1])
        elif object[0].lower() == "second stair":
            action = jump_second_stair(mario, object[1])
        elif object[0].lower() == "last stair":
            last_stair = True
        elif object[0].lower() == "goal":
            action = jump_goal(mario, object[1])

    print("action:", action)
    print("mario:", mario)
    print("object:", object)
    return action, last_stair

def jump_pipe(mario_location, pipe_location):
    distance = pipe_location[0] - mario_location[0]
    action = 3

    if distance > 0:
        if is_on_pipe(mario_location, pipe_location):
            action = 2

        elif distance <= 60:
            action = 2

            if distance <= 18:
                action = 6

                if distance >= 12:
                    for i in range(60):
                        action = 4

    return action


def jump_enemy(mario_location, enemy_location):
    distance = enemy_location[0] - mario_location[0]
    action = 3

    if distance > 0 and not is_below_enemy(mario_location, enemy_location):
        if distance <= 30:
            action = 2

    return action

def jump_hole(mario_location, hole_location):
    distance = hole_location[0] - mario_location[0]
    action = 3

    if distance > 0:
        if distance <= 40:
            action = 2

    return action

def jump_first_left_stair(mario_location, first_left_stair_location):
    distance = first_left_stair_location[0] - mario_location[0]
    action = 3

    if distance > 0:
        if distance <= 50:
            action = 4

        if distance <= 30 and is_on_ground(mario_location):
            action = 6

    return action

def jump_first_right_stair(mario_location, first_right_stair_location):
    distance = first_right_stair_location[0] - mario_location[0]
    action = 3

    if distance > 0:
        if distance <= 60:
            action = 4
    
            if distance <= 30 and is_on_ground(mario_location):
                action = 6

                if distance >= 45:
                    action = 2

    return action

def jump_second_stair(mario_location, second_stair_location):
    distance = second_stair_location[0] - mario_location[0]
    action = 3

    if distance > 0:
        if distance <= 60:
            action = 4

            if distance <= 25:
                action = 0

    return action

def jump_last_stair(env):
    for _ in range(10):
        env.step(0)
    for _ in range(4):
        for _ in range(15):
            env.step(2)
        for _ in range(10):
            env.step(0)
    
    obs, reward, terminated, truncated, info = env.step(0)

    return obs, reward, terminated, truncated, info

def jump_goal(mario_location, goal_location):
    distance = goal_location[0] - mario_location[0]
    action = 3

    if distance > 0:
        if distance <= 50:
            action = 4

            if distance <= 15 and is_on_ground(mario_location):
                action = 6

    return action
    

def is_on_pipe(mario_location, pipe_location):
    return mario_location[1] < pipe_location[1]

def is_below_enemy(mario_location, enemy_location):
    return mario_location[1] > enemy_location[1] + 25

def is_on_stair(mario_location, stair_location):
    return mario_location[1] < stair_location[1]

def is_on_ground(mario_location):
    return mario_location[1] >= 190

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs = None
done = True
env.reset()
for step in range(100000):
    last_stair = False
    if obs is not None:
        action, last_stair = make_action(obs, info, step, env, action)
    else:
        action = env.action_space.sample()
    
    if last_stair:
        obs, reward, terminated, truncated, info = jump_last_stair(env)
    else:
        obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    if done:
        env.reset()
env.close()
