"""Ground-truth Q/A pairs used by the Ragas evaluation harness.

Questions are grounded in *The Adventures of Sherlock Holmes* and *The
Memoirs of Sherlock Holmes*. One out-of-scope question is included to
probe the chatbot's refusal behaviour.
"""

from __future__ import annotations

EVALUATION_DATA: list[dict[str, str]] = [
    {
        "question": (
            "Who is Irene Adler and why is she important to Holmes?"
        ),
        "ground_truth": (
            "Irene Adler is an American opera singer who appears in "
            "'A Scandal in Bohemia'. She outwits Holmes when he tries to "
            "recover a compromising photograph for the King of Bohemia. "
            "She is notable as the only woman Holmes is shown to "
            "genuinely respect, and he afterwards refers to her simply "
            "as 'the woman'."
        ),
    },
    {
        "question": (
            "What was the Red-Headed League and what was really going on?"
        ),
        "ground_truth": (
            "The Red-Headed League was a bogus society that paid Jabez "
            "Wilson to copy out the Encyclopaedia Britannica for a few "
            "hours each day. Its real purpose was to keep Wilson out of "
            "his pawnshop so that his assistant, John Clay, and an "
            "accomplice could dig a tunnel from Wilson's cellar into the "
            "vault of the neighbouring City and Suburban Bank."
        ),
    },
    {
        "question": (
            "What method did Holmes use to find the photograph in "
            "A Scandal in Bohemia?"
        ),
        "ground_truth": (
            "Holmes staged a fake fire inside Irene Adler's house by "
            "throwing a smoke rocket through the window while Watson "
            "raised the alarm. He knew that in a fire a person instinctively "
            "saves their most precious possession, which revealed the "
            "photograph's hiding place."
        ),
    },
    {
        "question": (
            "Who is Dr. Watson and what is his relationship with Holmes?"
        ),
        "ground_truth": (
            "Dr. John H. Watson is a former army surgeon and the narrator "
            "of the stories. He shares lodgings with Sherlock Holmes at "
            "221B Baker Street, assists him on his cases, and records "
            "the adventures as the reader encounters them."
        ),
    },
    {
        "question": (
            "What was the speckled band and how did it kill?"
        ),
        "ground_truth": (
            "The 'speckled band' was a swamp adder, a venomous Indian "
            "snake that Dr. Grimesby Roylott trained to crawl through a "
            "ventilator and down a bell-rope into his stepdaughter's bed. "
            "Its bite killed Julia Stoner and was meant to kill Helen "
            "Stoner as well, until Holmes drove the snake back and it "
            "bit Roylott instead."
        ),
    },
    {
        "question": (
            "How did Neville St. Clair disappear?"
        ),
        "ground_truth": (
            "Neville St. Clair had been earning his living in secret as "
            "a professional beggar, Hugh Boone, in the City. He "
            "disappeared by removing his beggar's disguise at an opium "
            "den in Upper Swandam Lane. Holmes uncovered the deception "
            "by washing the grime from Boone's face, revealing St. Clair "
            "beneath."
        ),
    },
    {
        "question": (
            "What deductions did Holmes make from Mr. Jabez Wilson's "
            "appearance?"
        ),
        "ground_truth": (
            "From a glance, Holmes deduced that Wilson had done manual "
            "labour, taken snuff, been a Freemason, been in China, and "
            "had done a considerable amount of writing recently, based "
            "on clues such as his muscular right hand, an arc-and-"
            "compass breastpin, a Chinese-style tattoo, and the shiny "
            "cuff of his right sleeve."
        ),
    },
    {
        "question": (
            "What happens to Holmes at the Reichenbach Falls in "
            "The Final Problem?"
        ),
        "ground_truth": (
            "Holmes and Professor Moriarty meet at the Reichenbach Falls "
            "in Switzerland. After a struggle both men appear to plunge "
            "into the falls and Watson, finding only their footprints "
            "and Holmes's farewell note, believes that Holmes has been "
            "killed."
        ),
    },
    {
        "question": (
            "Who is Professor Moriarty and how does Holmes describe him?"
        ),
        "ground_truth": (
            "Professor James Moriarty is a mathematical genius turned "
            "criminal mastermind. Holmes describes him as the 'Napoleon "
            "of crime', the hidden organiser behind half the evil and "
            "nearly all the undetected crime in London."
        ),
    },
    {
        "question": "What is the capital of Germany?",
        "ground_truth": (
            "The Sherlock Holmes stories in the provided context do not "
            "contain information about the capital of Germany."
        ),
    },
]
