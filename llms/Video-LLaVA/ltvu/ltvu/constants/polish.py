SEP = '</s>'
REPLACE_PATTERNS = [  # the order matters
    {'replaced_with': '#C',
    'patterns': [
        r'[Tt]he (man|person)(?!(?:\s+in))',  # not followed by in
        r'[Tt]he (man|person) in (this|the) video',
        r'(?!\w)[Hh]e(?=\s)',
    ]},
    {'replaced_with': '',
    'patterns': [
        r'( [Tt]hese objects[^.]* |, which[^.]*| that[^.]*| to )?help(s|ing)?[^.]*answer[^.]*[\'\"].*[\'\"]',
        r'(?<=(, and ))\S+ is also using ',
    ]},
    {'replaced_with': '. And ',
    'patterns': [
        r'. \S+ is also using '
    ]},
    {'replaced_with': ' himself ',
    'patterns': [
        r' him '
    ]},
    {'replaced_with': 'A',
    'patterns': [
        r'[Ii]n this video, we can see a',
    ]},
    # {'replaced_with': 'the key objects would be',
    # 'patterns': [
    #     r'[Tt]he objects are',
    # ]}
]
