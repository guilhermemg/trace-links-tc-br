# Enumeration of the many data origins for the creation of the oracles

from enum import Enum

class DataOrigin(Enum):
    VOLUNTEERS = "VOLUNTEERS"
    EXPERT = "EXPERT"
    EXPERT_2 = "EXPERT_2"
    VOLUNTEERS_AND_EXPERT_UNION = "VOLUNTEERS_AND_EXPERT_UNION"
    VOLUNTEERS_AND_EXPERT_INTERSEC = "VOLUNTEERS_AND_EXPERT_INTERSEC"