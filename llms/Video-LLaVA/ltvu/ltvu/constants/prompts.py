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
        'Is there a checked shirt in this video?',

        # 'Did that appear in the video?'
    ]}
]

TEMPLATE_CAPTION_SEQUENCE = '{start_sec}s~{end_sec}s: {caption1}'
# ...
# 336.0 s ~ 337.5 s: The objects that help in answering the question "what color was the hammer on the work table?" are the hammer and the work table.</s>
# ...

MID_TERM_PROMPTS = [
    {'type': '20240219v0', 'prompts': [
        ''
        'Captions:\n{capseq}\n'
        ''
    ]}
]
