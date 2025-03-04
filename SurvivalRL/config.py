from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Simulation Environment Setting
    WINDOW_SIZE: int = 100
    GRID_SIZE: int = WINDOW_SIZE // 4
    MIN_TARGET_DISTANCE: int = 5

    # Debug setting
    DEBUG_MODE: bool = True
    PREDATOR: bool = True
    HERBIVORE: bool = True
    PLANT: bool = False

    # Animation Setting
    TARGET_FPS: int = 30
    DURATION: int = 5
    INTERVAL: float = 1000 / TARGET_FPS
    FRAMES: float = TARGET_FPS * DURATION