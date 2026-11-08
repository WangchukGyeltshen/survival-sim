"""Names for agents (for people watching; the agents themselves don't use them).

A name is made from syllables, chosen by a random generator seeded with the
agent's slot number and the tick it was born, so the same agent always gets
the same name, in the viewer, the event log, replays and the records.
"""

import random

SYLLABLES_A = ["ka", "mi", "ta", "lo", "ri", "sa", "no", "pe", "da", "yu", "ke", "bo", "ni", "ra",
               "zu", "ve", "ha", "jo", "li", "ma", "to", "se", "wa", "go"]
SYLLABLES_B = ["ra", "n", "vo", "la", "mi", "ko", "ta", "ri", "sh", "do", "na", "ki", "lu", "m",
               "re", "ya", "s", "to", "ba", "l"]


def identity_rng(slot, spawn_tick):
    return random.Random(int(slot) * 7919 + int(spawn_tick) * 104729)


def make_name(rnd):
    return (rnd.choice(SYLLABLES_A) + rnd.choice(SYLLABLES_B) +
            (rnd.choice(SYLLABLES_A) if rnd.random() < 0.4 else "")).capitalize()


def agent_name(slot, spawn_tick):
    return make_name(identity_rng(slot, spawn_tick))
