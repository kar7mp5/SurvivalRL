from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Simulation Environment Setting
    WINDOW_SIZE: int = 100
    GRID_SIZE: int = 20
    MIN_TARGET_DISTANCE: int = 5
    
    # Animation Setting
    TARGET_FPS: int = 30
    DURATION: int = 5
    INTERVAL: float = 1000 / TARGET_FPS
    FRAMES: float = TARGET_FPS * DURATION