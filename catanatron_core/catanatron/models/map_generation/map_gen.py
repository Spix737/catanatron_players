import pdb
import re
from idp_engine import IDP, model_expand

file_path = 'catanatron_core/catanatron/models/map_generation/catan_board_idp_theory.idp'
map_count = 0

# Function to split the content into sections
def split_into_sections(content):
    # Splitting by sections assuming they start with their names
    global map_count
    parts = {'vocabulary': '', 'theory': '', 'structure': ''}
    current_section = None
    for line in content.split('\n'):
        if line.startswith('vocabulary'):
            current_section = 'vocabulary'
        elif line.startswith('theory'):
            current_section = 'theory'
        elif line.startswith('structure'):
            current_section = 'structure'
        
        if current_section:
            parts[current_section] += line + '\n'
    return parts

def axialCoordToTileId(q, r):
    '''Converts axial coordinates to a tile id'''
    # If invalid, return null.
    if abs(q + r) > 2:
        return None
    idx = None
    if (r == -2):
        idx = abs(q)
    elif (r == -1): 
        idx = q + 4
    elif (r == 0): 
        idx = q + 9
    elif (r == 1): 
        idx = q + 14
    elif (r == 2): 
        idx = q + 18
    return idx

# Function to provide map (currently, balancing constraints must be amended on the idp file manually)
def generate_map():
    global map_count

    with open("catanatron_core/catanatron/models/map_generation/map_count.txt", 'r') as count_file:
        map_count_content = count_file.readlines()
        map_count = int(map_count_content[0].split(':')[1].strip())
        rest_of_content = map_count_content[1:]


    kb = IDP.from_file("catanatron_core/catanatron/models/map_generation/catan_board_idp_theory.idp")
    tiles = ['none'] * 19
    tokens = [None] * 19

    T, S = kb.get_blocks("T, S")
    for model in model_expand(T, S, max=1):
        # Split the model into categories
        categories = re.split(r"(\w+ := .+?})", model)
        del categories[0]   
        tile_type = categories[0]
        tile_token = categories[2]
        pattern = r"\((-?\d+), (-?\d+)\) -> (\w+)"
        # Create a dictionary to store the results
        matches = re.findall(pattern, tile_type)
        coordinate_tile_dict = {(int(x), int(y)): tile_type for x, y, tile_type in matches}
        matches = re.findall(pattern, tile_token)
        coordinate_token_dict = {(int(x), int(y)): tile_token for x, y, tile_token in matches}

        # Print the results
        for coordinates, tile_type in coordinate_tile_dict.items():
            q = coordinates[0]
            r = coordinates[1]
            idx = axialCoordToTileId(q, r)
            if idx is not None:
              tiles[idx] = tile_type
        for coordinates, tile_token in coordinate_token_dict.items():
            q = coordinates[0]
            r = coordinates[1]
            idx = axialCoordToTileId(q, r)
            if idx is not None:
              tokens[idx] = tile_token

    # Read the file contents
    with open(file_path, 'r') as file:
        original_content = file.read()
        sections = split_into_sections(original_content)
    with open(file_path, 'w') as file:
        new_map_constraint = f"// map_id: {map_count+1}\n"
        
        index = 0
        for axial_coord, tile_value in coordinate_tile_dict.items():
            if index == 0:
                new_map_constraint += "~(tile_type"+str(axial_coord)+" = "+tile_value+""
                index+=1
            else:
                new_map_constraint += " & tile_type"+str(axial_coord)+" = "+tile_value+""
        for axial_coord, token_value in coordinate_token_dict.items():
            new_map_constraint += " & tile_token"+str(axial_coord)+" = "+token_value+""
        new_map_constraint += ").\n"
    
    theory_lines = sections['theory'].rstrip()
    if theory_lines.endswith("}"):
        # Replace the last closing brace with the new_map_constraint followed by a closing brace
        sections['theory'] = f"{theory_lines[:-1].rstrip()}\n{new_map_constraint}}}\n\n"

    updated_content = sections['vocabulary'] + sections['theory'] + sections['structure']
    with open(file_path, 'w') as file:
        file.write(updated_content)
    rest_of_content += "tiles:"+str(tiles)+"\n"+"tokens:"+str(tokens)+"\n"
    with open("catanatron_core/catanatron/models/map_generation/map_count.txt", 'w') as count_file:
        count_file.write(f"map_count:{map_count+1}\n"+("").join(rest_of_content))

    return tiles, tokens


for i in range(10000):
    generate_map()
    if i%100 == 0:
        print(f"Map {i+1} generated")