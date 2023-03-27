from dataclasses import dataclass


@dataclass
class PermissiveFilterConfig:
    max_ll: int = 1200
    min_len: int = 8
    min_max_ll: int = 5
    max_size_kbs: int = 1_000
    min_alphanum: float = 0.0001
    max_alphanum: float = 0.9900
    min_ave_ll: int = 8
    min_lines: int = 2


@dataclass
class ModerateFilterConfig:
    max_ll: int = 600
    min_len: int = 50
    min_max_ll: int = 25
    max_size_kbs: int = 750
    min_alphanum: float = 0.001
    max_alphanum: float = 0.975
    min_ave_ll: int = 16
    min_lines: int = 3



@dataclass
class AggressiveFilterConfig:
    max_ll: int = 300
    min_len: int = 100
    min_max_ll: int = 32
    max_size_kbs: int = 500
    min_alphanum: float = 0.0025
    max_alphanum: float = 0.9700
    min_ave_ll: int = 20
    min_lines: int = 5
