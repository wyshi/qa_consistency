
RAW_BLENDER_PATH = "~/ParlAI/data/models/blender/blender_1Bdistill/model"
MAX_CANDIDATE = 11

NO_ANSWER_TOKEN = '[no_answer]'

REPETITIVE = 'repetitive'
CONTRADICTION = 'contradiction'
GOOD = 'none_all_good'

REPLY_TO_HUMAN_OFFENSIVE_MSG = "That message seems offensive. Please don't say it again. Ending chat now."

def convert_prefix(sent):
    """
    >>> Counter([d[0][0].split(":")[0] for d in data] + [d[0][1].split(":")[0] for d in data])
    Counter({'M': 1241, 'W': 1119, 'F': 122, 'Man': 15, 'Woman': 15, 
    'A': 6, 'B': 6, 'Customer': 3, 'Caller': 2, 'Dave': 2, 'Mary': 1, 
    'Carla': 1, 'Apartment Owner': 1, 'W1': 1, 'Guest': 1, 'Narrator': 1, 
    'Presenter': 1, 'Maria': 1, 'Bank Teller': 1, 'Ted': 1, 'Sales Associate': 1, 
    "Dave's Sister": 1, 'Ed': 1, 'Wife': 1, 'Security': 1, 'Stuart': 1, 'Mark': 1, 
    'Store Employee': 1, 'Girl': 1, 'Delia Robinson': 1, 'Kelly': 1, 'News Reporter': 1, 
    'Mall': 1, 'Travel Agent': 1, 'Rental Car Agent': 1, 'David': 1, 'Mad': 1, 'Steve': 1, 
    'W2': 1, 'Hotel Clerk': 1, 'Captain': 1, 'John Knox': 1, 'Robber': 1, 'Susan': 1, 
    'Rocky': 1, 'Husband': 1, 'Passenger': 1, 'Telemarketer': 1, 'Amy': 1, 'Jenny': 1, 
    'Mr. Taylor': 1, 'Dad': 1, 'Mr. Dong': 1, 'Mr. Adams': 1, 'Young Girl': 1})

    """
    prefix_map = {"W:": "Woman:", "F:": "Female:", "M:": "Man:"}

    for k, v in prefix_map.items():
        if sent.startswith(k):
            sent = v + sent[2:]

    return sent


def generate_document(dialogue, question):
    return f"{question} \\n {dialogue}"

