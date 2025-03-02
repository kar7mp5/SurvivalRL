from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    WINDOW_SIZE: int = 40
    GRID_SIZE: int = 10
    MIN_TARGET_DISTANCE: int = 5