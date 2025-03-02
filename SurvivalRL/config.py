from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    WINDOW_SIZE: int = 20
    GRID_SIZE: int = 5
    MIN_TARGET_DISTANCE: int = 5