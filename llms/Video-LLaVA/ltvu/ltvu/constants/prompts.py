SHORT_TERM_PROMPTS = [
    {'type': 'query_agnostic', 'prompts': [
        'What would be key objects in this video?',

        'Rate your answer by yourself in a confidence level of 1 ~ 5.'
    ]},

    {'type': 'query_simple', 'prompts': [
        'Natural Language Queries (NLQ) is a task that involves processing and understanding queries expressed in '
        'natural language(e.g., “What did I put in the drawer?”) to accurately localize and identify a specific '
        'temporal interval within a video where the answer to the query can be deduced or is explicitly visible.\n'
        'The video you are currently viewing is a part of the entire video. Given the query {query} for this video, '
        'generate a caption based on the key object that would be most helpful in solving the NLQ task.',

        'Rate your previous answer by yourself in a confidence level of 1 ~ 5.'
    ]},

    {'type': 'query_verbose', 'prompts': [
        'Natural Language Queries (NLQ) is a task that involves processing and understanding queries expressed in '
        'natural language(e.g., “What did I put in the drawer?”) to accurately localize and identify a specific '
        'temporal interval within a video where the answer to the query can be deduced or is explicitly visible.\n'
        'The current NLQ task involves videos that are very long, but the section containing the answer is very '
        'short, making it difficult to solve the problem. Therefore, we plan to divide the video into multiple '
        'short clips, find the key object that is most crucial within these clips, and generate captions containing '
        'this key object. Then, based on these captions, we aim to solve the NLQ task.\n'
        'The video you are currently viewing is a part of the entire video. Given the query {query} for this '
        'video, generate a caption based on the key object that would be most helpful in solving the NLQ task.',

        'Rate your previous answer by yourself in a confidence level of 1 ~ 5.'
    ]},

    {'type': '20240219v0', 'prompts': [
        'What can you see in this video?',
        'What objects are there that help in answering the question "{query}"?',
    ]},

    {'type': '20240301v1', 'prompts': [
        '{query}',
        'Did that appear in the video?'
    ]},

    {'type': '20240304v1', 'prompts': [
        'Describe the video.',
        'Can you answer "{query}"?'
    ]},

    {'type': '20240307v1', 'prompts': [
        # 그나마 잘 되는 듯
        'Describe the video in a single sentence.',
        'Can you answer "{query}"?'
    ]},

    {'type': '20240308v1', 'prompts': [
        'Can you answer "{query}"?'
    ]},
    {'type': '20240308v2', 'prompts': [
        'Describe the video within 3 sentences.',
        'Can you answer "{query}"?'
    ]},
    {'type': '20240308v3', 'prompts': [
        # 아무 답이나 하려고 함
        'Describe the video within 3 sentences.',
        'What do you think a random person would answer the qustion "{query}" after watching this video?'
    ]},
    {'type': '20240308v4', 'prompts': [
        # 'Can you answer the question "{query}"?',
        # 'But I cannot find it in this video. Are you sure?',  # 이거 살짝 되는 듯? 근데 yes/no가 안 나옴
        'What am I doing?',
        'Are there only things in this video that are unhelpful for answering this question "{query}"?',
    ]},
]

assert len(SHORT_TERM_PROMPTS) == len(set([x['type'] for x in SHORT_TERM_PROMPTS])), 'type names are not unique'
