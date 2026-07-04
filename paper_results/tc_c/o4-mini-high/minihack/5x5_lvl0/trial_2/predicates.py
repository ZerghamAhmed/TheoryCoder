def descended(state, agent, stair):
    """
    Returns True if the agent is standing on the same coordinates as the stair.

    Parameters:
    - state: dict containing object positions, e.g.
        {
          'staircase_up': [[36, 11]],
          'human_rogue_called_agent': [[38, 8]],
          'staircase_down': [[40, 7]],
          'inventory': [],
          'won': False,
          'lost': False
        }
    - agent: string name of the agent object (e.g., 'human_rogue_called_agent')
    - stair: string name of the stair object (e.g., 'staircase_down')

    Returns:
    - bool: True if agent's [x,y] equals stair's [x,y], False otherwise.
    """
    # Get the position lists
    agent_pos_list = state.get(agent)
    stair_pos_list = state.get(stair)

    # Both must exist and contain at least one [x,y]
    if not agent_pos_list or not stair_pos_list:
        return False

    # Unpack coordinates
    ax, ay = agent_pos_list[0]
    sx, sy = stair_pos_list[0]

    return (ax == sx) and (ay == sy)