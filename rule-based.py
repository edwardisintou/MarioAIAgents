from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string

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
        # "block": ["Images/block1.png", "Images/block2.png", "Images/block3.png", "Images/block4.png"],
        "block": ["Images/block2.png"],
        # "question_block": ["Images/questionA.png", "Images/questionB.png", "Images/questionC.png"],
        # "hole": ["Images/new_hole.png"],
        # "hole4": ["Images/hole4.png"],
        # "hole_resize": ["Images/hole5.png"],
    },
    # "hole": {
    #     "hole": ["Images/hole.png"]
    # },
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

    block_set = set()
    block_locations = object_locations["block"]
    # print("block locations:", block_locations)
    # print("len of blocks:", len(block_locations))
    hole_position = None
    if len(block_locations) > 0:
        for block in block_locations:
            block_position_x = block[0][0]

            if block_position_x > mario_position[0]:
                block_set.add(block_position_x)

        print("mario: ", mario_position[0])
        print("current blocks:", block_set)

        current_block = block_locations[0][0]
        current_block_x = current_block[0]

        if current_block_x > mario_position[0]:
            next_block_x = current_block_x + 16
            print("next_block_x", next_block_x)
        
            if next_block_x not in block_set:
                hole_position = current_block
                print("hole")

    next_object = nearest_object(mario_position, enemy_position, pipe_position, hole_position)

    if enemy_position == next_object:
        return ["enemy", enemy_position]
    elif pipe_position == next_object:
        return ["pipe", pipe_position]
    elif hole_position == next_object:
        return ["hole", hole_position]
    else:
        return None


def nearest_object(mario_position, enemy_position, pipe_position, hole_position):
    objects = []
    locations = []

    for object in enemy_position, pipe_position, hole_position:
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

    if object is None:
        action = 3
    else:
        if object[0].lower() == "pipe":
            action = jump_pipe(mario, object[1])
        elif object[0].lower() == "enemy":
            action = jump_enemy(mario, object[1])
        elif object[0].lower() == "hole":
            action = jump_hole(mario, object[1])

        # print("distance:", abs(mario[0] - object[1][0]))

    # print("action:", action)
    # print("mario:", mario)
    # print("object:", object)
    return action

def jump_hole(mario_location, hole_location):
    distance = hole_location[0] - mario_location[0]
    action = 3

    if distance > 0:
        if distance <= 150:
            action = 2

    return action


def jump_pipe(mario_location, pipe_location):
    distance = pipe_location[0] - mario_location[0]
    action = 3

    if distance > 0:
        if is_on_pipe(mario_location, pipe_location):
            action = 2

        elif distance <= 58:
            action = 2

            if distance <= 20:
                action = 6

                if distance >= 15:
                    for i in range(40):
                        action = 4

    return action


def jump_enemy(mario_location, enemy_location):
    distance = enemy_location[0] - mario_location[0]
    action = 3

    if distance > 0:
        if distance <= 40:
            action = 2

            if distance <= 20:
                action = 6

    return action

def is_on_pipe(mario_location, pipe_location):
    mario_y = mario_location[1]
    pipe_y = pipe_location[1]

    return mario_y < pipe_y

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs = None
done = True
env.reset()
for step in range(100000):
    if obs is not None:
        action = make_action(obs, info, step, env, action)
    else:
        action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        env.reset()
env.close()