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
        map_count_content = count_file.read()
        for line in map_count_content.split('\n'):
            map_count = int(line.split(':')[1].strip())

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

    # print('tiles: ',tiles)
    # print('tokens: ',tokens)

    # Read the file contents
    with open(file_path, 'r') as file:
        original_content = file.read()
        sections = split_into_sections(original_content)
    with open(file_path, 'w') as file:

        # Each of the three sections ends with a line that contains "}", to signify the end of the section
        # The new constraints generated below need to be added before that closing "}" tag, which should be the last line of the section
        new_map_constraint = f"\n// map_id: {map_count+1}"
        
        # Amend the theory section as per the condition
        new_map_constraint += """
// Constraint to exclude a specific board configuration
!q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19 in Q, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19 in R:"""

        for i in range(19):
            if i == 0:
                new_map_constraint += "(tile_type(q1, r1) = "+tiles[i]+" & tile_token(q1, r1) = "+tokens[i]+") & "
            elif i == 18:
                new_map_constraint += "(tile_type(q19, r19) = "+tiles[i]+" & tile_token(q19, r19) = "+tokens[i]+").\n"
            else:
                index = str(i)
                new_map_constraint += "(tile_type(q"+index+", r"+index+") = "+tiles[i]+" & tile_token(q"+index+", r"+index+") = "+tokens[i]+") & "
        
        # Combine the sections back into one string
        # THIS LINE WILL NOT WORK AS IT WILL ADD AFTER THE THEORY'S "}" TAG
    
    # import pdb
    # pdb.set_trace()
    
    theory_lines = sections['theory'].rstrip()
    if theory_lines.endswith("}"):
        # Replace the last closing brace with the new_map_constraint followed by a closing brace
        sections['theory'] = f"{theory_lines[:-1].rstrip()}\n{new_map_constraint}}}\n\n"
    # sections['theory'] = '\n'.join(theory_lines)

    updated_content = sections['vocabulary'] + sections['theory'] + sections['structure']
    with open(file_path, 'w') as file:
        file.write(updated_content)
    with open("catanatron_core/catanatron/models/map_generation/map_count.txt", 'w') as count_file:
        count_file.write(f"map_count:{map_count+1}\n")

    return tiles, tokens


map_tiles, map_tokens = generate_map()
print("tiles: ", map_tiles)
print("tokens: ", map_tokens)


