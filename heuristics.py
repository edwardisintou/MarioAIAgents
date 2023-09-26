from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string
import rules as rules
import random

# code for locating objects on the screen in super mario bros
# by Lauren Gee

# Template matching is based on this tutorial:
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

################################################################################

# change these values if you want more/less printing
PRINT_GRID = False
PRINT_LOCATIONS = False

# If printing the grid doesn't display in an understandable way, change the
# settings of your terminal (or anaconda prompt) to have a smaller font size,
# so that everything fits on the screen. Also, use a large terminal window /
# whole screen.

# other constants (don't change these)
SCREEN_HEIGHT = 240
SCREEN_WIDTH = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# You can add more images to improve the object locator, so that it can locate
# more things. For best results, paint around the object with the exact shade of
# blue as the sky colour. (see the given images as examples)
#
# Put your image filenames in image_files below, following the same format, and
# it should work fine.

# filenames for object templates
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
        "tall": ["Images/tall_marioA.png", "Images/tall_marioB.png", "Images/tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["Images/goomba.png"],
        "koopa": ["Images/koopaA.png", "Images/koopaB.png"],
    },
    "block": {
        "block": ["Images/block1.png", "Images/block2.png", "Images/block3.png", "Images/block4.png"],
        "question_block": ["Images/questionA.png", "Images/questionB.png", "Images/questionC.png"],
        "pipe": ["Images/pipe_upper_section.png", "Images/pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["Images/mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.
        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    },
}


def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0] * image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None  # this is important for avoiding a problem where some things match everything
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


# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}

# generate all templatees
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

################################################################################
# PRINTING THE GRID (for debug purposes)

colour_map = {
    (104, 136, 252): " ",  # sky blue colour
    (0, 0, 0): " ",  # black
    (252, 252, 252): "'",  # white / cloud colour
    (248, 56, 0): "M",  # red / mario colour
    (228, 92, 16): "%",  # brown enemy / block colour
}
unused_letters = sorted(set(string.ascii_uppercase) - set(colour_map.values()), reverse=True)
DEFAULT_LETTER = "?"


def _get_colour(colour):  # colour must be 3 ints
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]

    # if we haven't seen this colour before, pick a letter to represent it
    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER


def print_grid(obs, object_locations):
    pixels = {}
    # build the outlines of located objects
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

    # print the screen to terminal
    print("-" * SCREEN_WIDTH)
    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                # this pixel is part of an outline of an object,
                # so use that instead of the normal colour symbol
                colour = pixels[coords]
            else:
                # get the colour symbol for this colour
                colour = _get_colour(obs[y][x])
            line.append(colour)
        print("".join(line))


################################################################################
# LOCATING OBJECTS


def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break

    #      [((x,y), (width,height))]
    return [(loc, locations[loc]) for loc in locations]


def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))

    # stop early if there are no pipes
    if not upper_locs:
        return []

    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
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
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue

            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations


################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

consecutive_left_moves = 0  # Track consecutive "move left" actions


def make_action(screen, info, step, env, prev_action):
    # print(screen)
    # print(info)
    # print(step)
    # print(env)
    # print(prev_action)
    # print(SIMPLE_MOVEMENT[prev_action])
    mario_status = info["status"]
    # print(mario_status)
    object_locations = locate_objects(screen, mario_status)
    # print(object_locations)
    # print('\n')

    # You probably don't want to print everything I am printing when you run
    # your code, because printing slows things down, and it puts a LOT of
    # information in your terminal.

    # Printing the whole grid is slow, so I am only printing it occasionally,
    # and I'm only printing it for debug purposes, to see if I'm locating objects
    # correctly.
    if PRINT_GRID and step % 100 == 0:
        print_grid(screen, object_locations)
        # If printing the grid doesn't display in an understandable way, change
        # the settings of your terminal (or anaconda prompt) to have a smaller
        # font size, so that everything fits on the screen. Also, use a large
        # terminal window / whole screen.

        # object_locations contains the locations of all the objects we found
        print(object_locations)

    # List of locations of Mario:
    mario_locations = object_locations["mario"]
    # print("mario_locations")
    # print(mario_locations)
    # print(mario_locations[0][0][0])
    # (There's usually 1 item in mario_locations, but there could be 0 if we
    # couldn't find Mario. There might even be more than one item in the list,
    # but if that happens they are probably approximately the same location.)

    # List of locations of enemies, such as goombas and koopas:
    enemy_locations = object_locations["enemy"]
    # print("enemy_locations")
    # print(enemy_locations)
    # print('\n')

    # List of locations of blocks, pipes, etc:
    block_locations = object_locations["block"]
    # print(block_locations)
    # print('\n')

    # List of locations of items: (so far, it only finds mushrooms)
    item_locations = object_locations["item"]

    # This is the format of the lists of locations:
    # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
    #
    # x_coordinate and y_coordinate are the top left corner of the object
    #
    # For example, the enemy_locations list might look like this:
    # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]

    if PRINT_LOCATIONS:
        # To get the information out of a list:
        for enemy in enemy_locations:
            enemy_location, enemy_dimensions, enemy_name = enemy
            x, y = enemy_location
            width, height = enemy_dimensions
            print("enemy:", x, y, width, height, enemy_name)

        # Or you could do it this way:
        for block in block_locations:
            block_x = block[0][0]
            block_y = block[0][1]
            block_width = block[1][0]
            block_height = block[1][1]
            block_name = block[2]
            print(f"{block_name}: {(block_x, block_y)}), {(block_width, block_height)}")

        # Or you could do it this way:
        for item_location, item_dimensions, item_name in item_locations:
            x, y = item_location
            print(item_name, x, y)

        # gym-super-mario-bros also gives us some info that might be useful
        print(info)
        # see https://pypi.org/project/gym-super-mario-bros/ for explanations

        # The x and y coordinates in object_locations are screen coordinates.
        # Top left corner of screen is (0, 0), top right corner is (255, 0).
        # Here's how you can get Mario's screen coordinates:
        if mario_locations:
            location, dimensions, object_name = mario_locations[0]
            mario_x, mario_y = location
            print("Mario's location on screen:", mario_x, mario_y, f"({object_name} mario)")

        # The x and y coordinates in info are world coordinates.
        # They tell you where Mario is in the game, not his screen position.
        mario_world_x = info["x_pos"]
        mario_world_y = info["y_pos"]
        # Also, you can get Mario's status (small, tall, fireball) from info too.
        mario_status = info["status"]
        print("Mario's location in world:", mario_world_x, mario_world_y, f"({mario_status} mario)")

    ################################################################################

    # TODO: Write code for a strategy, such as a rule based agent.

    # Choose an action from the list of available actions.
    # For example, action = 0 means do nothing
    #              action = 1 means press 'right' button
    #              action = 2 means press 'right' and 'A' buttons at the same time
    #              action = 3 means press 'right and 'B'
    #              action = 4 means press 'right' and 'A' and 'B'
    #              action = 5 means press 'A' button
    #              action = 6 means press 'left' button

    if enemy_locations:
        if any(block[2] == "pipe" for block in block_locations):
            # Calculate the horizontal distance between Mario and the closest enemy
            mario_x = mario_locations[0][0][0]
            closest_enemy_x = min(enemy[0][0] for enemy in enemy_locations)
            distance_to_closest_enemy = abs(mario_x - closest_enemy_x)

            # Calculate the horizontal distance between Mario and the closest pipe
            closest_pipe_x = min(pipe[0][0] for pipe in block_locations if pipe[2] == "pipe")
            distance_to_closest_pipe = abs(mario_x - closest_pipe_x)

            if distance_to_closest_pipe < distance_to_closest_enemy:
                # print("here")
                action = pipe_pass(prev_action, mario_locations, block_locations)
                return action
            else:  # enemy is closer
                # print("ignore 4 now")
                action = near_enemy(mario_locations, closest_enemy_x)
                return action
                # action = pipe_pass(prev_action, mario_locations, block_locations)
                # return action

        else:
            action = near_enemy_right(mario_locations, enemy_locations)
            # print("B")
            return action

    elif any(block[2] == "pipe" for block in block_locations):
        action = pipe_pass(prev_action, mario_locations, block_locations)
        # print("A")
        print(action)
        return action

    else:
        # With a random agent, I found that choosing the same random action
        # 10 times in a row leads to slightly better performance than choosing
        # a new random action every step.

        # return prev_action
        action = 1
        # print(action)
        return action


################################################################################
# FUNCTIONS


def near_enemy_right(mario_locations, enemy_locations, threshold=58):
    """Consider Multiple Enemies: If there are multiple enemies on the screen,
    you might want to consider the closest enemy rather than just the first one
    in the list. Calculate the distance to all enemies and choose the action
    based on the closest one."""
    mario_ingame_x = mario_locations[0][0][0]

    # Check if there are any enemy locations
    if not enemy_locations:
        return 1  # No enemies nearby

    # Calculate the horizontal distance between Mario and the enemy
    x_distance = mario_ingame_x - enemy_locations[0][0][0]
    x_distance = abs(x_distance)
    # print(x_distance)
    # print('\n')

    # Check if the enemy is to the right of Mario and within the threshold distance
    if x_distance > 53 and x_distance < threshold:
        return 2
    # elif

    return 1


def near_enemy(mario_locations, closest_enemy_x, threshold=20):
    mario_ingame_x = mario_locations[0][0][0]

    # Calculate the horizontal distance between Mario and the enemy
    x_distance = mario_ingame_x - closest_enemy_x
    x_distance = abs(x_distance)
    print(x_distance)
    print("\n")

    # Check if the enemy is to the right of Mario and within the threshold distance
    if x_distance > 0 and x_distance < threshold:
        return 5  # Press 'A' button (jump) to avoid the enemy

    return 1


def pipe_pass(prev_action, mario_locations, block_locations):
    global consecutive_left_moves  # Use a global variable
    # if prev_action in [1,2,5]:
    #     action = jump_on_air(mario_locations, block_locations)
    #     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #     return action
    # else:
    for block in block_locations:
        block_x = block[0][0]
        block_y = block[0][1]
        block_width = block[1][0]
        block_height = block[1][1]
        block_name = block[2]

        if block_name == "pipe":
            mario_ingame_x = mario_locations[0][0][0]
            mario_ingame_y = mario_locations[0][0][1]
            mario_width = mario_locations[0][1][0]
            mario_height = mario_locations[0][1][1]

            # print("##########")
            # print(mario_ingame_x)
            # print(mario_ingame_y)
            # print(block_x)
            # print(block_y)
            # print('\n')
            # print("##########^^^")

            # Check if Mario is near the pipe horizontally
            if consecutive_left_moves:
                consecutive_left_moves -= 1
                # print("great success")
                return 6

            if not_too_close_to_pipe(mario_ingame_x, mario_width, block_x):
                # Check if Mario can jump over the pipe
                if can_jump_over_pipe(mario_ingame_x, mario_ingame_y, block_x, block_y):
                    if prev_action == 5:
                        print("abc")
                        print(block_x)
                        print(mario_ingame_x)
                        print("\n")
                        return random.choice([1, 2])
                    else:
                        print("def")
                        print(block_x)
                        print(mario_ingame_x)
                        print("\n")
                        return 5  # Jump
                    # else:
                    #     return random.choice([1, 2])  # Move right or jump-right
                else:
                    # # Choose an appropriate action based on Mario's position relative to the pipe
                    # if mario_ingame_y < block_y:
                    #     return 5  # Jump to avoid hitting the pipe from below
                    # elif mario_ingame_y > block_y + block_height:
                    #     return 1  # Move right if Mario is below the pipe
                    # else:
                    print("ghi")
                    return random.choice([1, 2, 5])

            else:  # if mario goes left on pipe,increase 12
                if (
                    not (is_on_top_of_pipe(mario_ingame_x, mario_ingame_y, mario_width, block_x, block_y, block_width))
                    and (mario_ingame_x < (block_x - 12))
                    and (mario_ingame_x > (block_x - 20))
                ):
                    # if not(is_on_top_of_pipe(mario_ingame_x, mario_ingame_y, mario_width, block_x, block_y, block_width)) and (not_too_close_to_pipe(mario_ingame_x, mario_width, block_x) == False):
                    print("jkl")
                    print(is_on_top_of_pipe(mario_ingame_x, mario_ingame_y, mario_width, block_x, block_y, block_width))
                    print("block_x")
                    print(block_x)
                    print("block_y")
                    print(block_y)
                    print("mario_ingame_x")
                    print(mario_ingame_x)
                    print("mario_ingame_y")
                    print(mario_ingame_y)
                    print("block_width")
                    print(block_width)
                    print("\n")
                    consecutive_left_moves = 0
                    consecutive_left_moves += 30
                    return 6
                else:
                    print("mno")
                    return random.choice([1, 2, 5])

        else:
            continue


# def tall_pipe_positioning()


def not_too_close_to_pipe(mario_ingame_x, mario_width, block_x, threshold=17):
    print("not_too_close_ti_pipe")
    print(mario_ingame_x)
    print(block_x)
    print(threshold)
    return threshold <= (mario_ingame_x + mario_width) <= block_x


# def not_too_close_to_pipe(mario_ingame_x, mario_width, block_x, block_width, min_distance=5):
#     """
#     Check if Mario is not too close to the pipe horizontally.

#     Args:
#         mario_ingame_x (int): Mario's x-coordinate.
#         mario_width (int): Width of Mario's bounding box.
#         block_x (int): The x-coordinate of the pipe.
#         block_width (int): Width of the pipe.
#         min_distance (int): The minimum acceptable horizontal distance.

#     Returns:
#         bool: True if Mario is not too close to the pipe, False otherwise.
#     """
#     # Calculate the horizontal distance between the centers of Mario and the pipe
#     mario_center_x = mario_ingame_x + mario_width // 2
#     pipe_center_x = block_x + block_width // 2
#     distance = abs(mario_center_x - pipe_center_x)

#     # Check if the calculated distance is greater than or equal to the minimum distance
#     return distance >= min_distance


def can_jump_over_pipe(mario_ingame_x, mario_ingame_y, block_x, block_y):
    """
    Determine if Mario can jump over the pipe.

    Args:
        mario_ingame_x (int): Mario's x-coordinate.
        mario_ingame_y (int): Mario's y-coordinate.
        block_x (int): The x-coordinate of the pipe.
        block_y (int): The y-coordinate of the pipe.

    Returns:
        bool: True if Mario can jump over the pipe, False otherwise.
    """
    # Define Mario's jump ability (you can adjust this)
    mario_jump_height = 35  # Example jump height

    # Calculate the vertical distance between Mario and the pipe
    vertical_distance = abs(mario_ingame_y - block_y)

    # Check if Mario is close to the pipe horizontally and can jump high enough
    if (mario_ingame_x >= block_x - 40) and (vertical_distance <= mario_jump_height):
        return True
    else:
        return False


def is_on_top_of_pipe(mario_ingame_x, mario_ingame_y, mario_width, block_x, block_y, block_width):
    # Define a threshold to determine if Mario is on top of a pipe
    threshold = 0  # Adjust this value as needed

    # Check if Mario's x-coordinate is within the horizontal range of the block
    if block_x <= mario_ingame_x + mario_width <= block_x + block_width:
        # Check if Mario's y-coordinate is close to the top of the block
        if (mario_ingame_y - block_y) < 0:
            print("mario_ingame_y - block_y")
            print(mario_ingame_y)
            print(block_y)
            print((mario_ingame_y - block_y))
            return True

    return False


def on_air_peak(mario_locations, block_locations):
    # to find where mario is x wise. ex: if on pipe, block_y will be lower than if on block
    # will store the y of the object mario is standing on
    min_block_y = 9999999

    mario_ingame_x = mario_locations[0][0][0]
    mario_ingame_y = mario_locations[0][0][1]

    for block in block_locations:
        block_x = block[0][0]
        block_y = block[0][1]
        block_width = block[1][0]
        block_height = block[1][1]
        block_name = block[2]

        x_loc = block_x + block_width

        if block_x < mario_ingame_x <= x_loc:
            # print("on_air_peak")
            # print(block_name)
            # print(block_y)

            if min_block_y <= block_y:
                continue
            else:
                min_block_y = block_y

            # print("min_block_y")
            # print(min_block_y)

    # print("min_block_y - mario_ingame_y")
    if (min_block_y - mario_ingame_y) > 30:
        return True
    else:
        return False


def jump_on_air(mario_locations, block_locations):
    if on_air_peak(mario_locations, block_locations):
        print("got here")
        return 5
    else:
        return random.choice([1, 2])


################################################################################

env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

stuck = 0

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