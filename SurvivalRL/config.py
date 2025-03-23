from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Simulation Environment Setting
    WINDOW_SIZE: int = 50
    GRID_SIZE: int = WINDOW_SIZE // 10
    MIN_TARGET_DISTANCE: int = 5

    # Number of creatures
    HERBI_NUM: int = 4
    PRED_NUM: int = 1
    PLANT_NUM: int = 10

    # Text setting
    DEFAULT_FONT_SIZE: int = 7
    DEBUG_FONT_SIZE: int = 6
    
    # Debug setting
    DEBUG_MODE: bool = False
    PREDATOR: bool = False
    HERBIVORE: bool = True
    PLANT: bool = False

    # Animation Setting
    TARGET_FPS: int = 30
    DURATION: int = 30
    INTERVAL: float = 1000 / TARGET_FPS
    FRAMES: float = TARGET_FPS * DURATION