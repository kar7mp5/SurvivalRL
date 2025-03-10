from collections import defaultdict
from SurvivalRL import Config
import numpy as np
import matplotlib.patches as patches


class SpatialHashGrid:
    """
    Efficient spatial partitioning using a hash grid for fast collision detection.
    """
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}

    def _get_cell_key(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, obj):
        """Insert object into the correct grid cell."""
        cell_key = self._get_cell_key(obj.pos.x, obj.pos.y)
        if cell_key not in self.grid:
            self.grid[cell_key] = []
        self.grid[cell_key].append(obj)

    def retrieve_nearby(self, obj):
        """Retrieve nearby objects for collision checking."""
        cell_x, cell_y = self._get_cell_key(obj.pos.x, obj.pos.y)
        possible_objects = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in self.grid:
                    possible_objects.extend(self.grid[cell_key])
        return possible_objects

    def retrieve_in_fov_range(self, x, y, fov_radius):
        """Retrieve all objects within a given FOV radius using an adaptive cell range."""
        cell_x, cell_y = self._get_cell_key(x, y)
        search_radius = int(np.ceil(fov_radius / self.cell_size))  # 검색할 셀 개수

        possible_objects = []
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in self.grid:
                    possible_objects.extend(self.grid[cell_key])

        return possible_objects

    def clear(self):
        """Clear the spatial hash grid (for each frame update)."""
        self.grid = {}


class GameObject:
    """ 
    Manages all objects in the game using Spatial Hash Grid.
    """

    def __init__(self, ax, cell_size=Config.GRID_SIZE):
        """
        Initializes the GameObject manager.

        Args:
            ax (matplotlib.axes.Axes): The axis where objects will be drawn and managed.
            cell_size (int): Size of each cell in the spatial hash grid.
        """
        self.ax = ax
        self.objects = []
        self.spatial_grid = SpatialHashGrid(cell_size)
        self.grid_patches = [] 

    def add_object(self, obj):
        """
        Adds an object to the game and draws it on the axis.

        Args:
            obj (Obj): An instance of a game object (e.g., Circle, Rectangle).
        """
        obj.draw()
        self.objects.append(obj)

    def update(self, fps):
        """
        Updates all objects in the game by calling their respective update methods.

        Returns:
            list: A list of updated shapes for animation rendering.
        """
        # Spatial Hash Grid 초기화
        self.spatial_grid.clear()

        # 모든 객체를 Spatial Hash Grid에 삽입
        for obj in self.objects:
            self.spatial_grid.insert(obj)

        # 모든 객체 업데이트
        for obj in self.objects:
            obj.update(fps, self.spatial_grid)

        self.draw_grid()

        return [obj.shape for obj in self.objects]

    def draw_grid(self):
        """ 
        Draws the spatial hash grid on the figure, highlighting active cells and FOV-affected cells.
        """
        # 기존의 그리드 박스 제거
        for patch in self.grid_patches:
            patch.remove()
        self.grid_patches.clear()

        # 현재 활성화된 그리드 셀과 FOV 내 검색된 셀 저장
        fov_cells = set()
        object_cells = set()

        # 1️⃣ 객체가 포함된 셀 저장
        for obj in self.objects:
            cell_x, cell_y = self.spatial_grid._get_cell_key(obj.pos.x, obj.pos.y)
            object_cells.add((cell_x, cell_y))

            # 2️⃣ FOV 반경 내 모든 셀 추가 (FOV 속성이 존재하는 경우만)
            if getattr(obj, "FOV_RADIUS", None):  # 안전하게 FOV_RADIUS가 존재하는지 확인
                fov_radius = obj.FOV_RADIUS
                search_radius = int(np.ceil(fov_radius / self.spatial_grid.cell_size))

                for dx in range(-search_radius, search_radius + 1):
                    for dy in range(-search_radius, search_radius + 1):
                        fov_cell_x, fov_cell_y = cell_x + dx, cell_y + dy

                        # 3️⃣ FOV 범위 내인지 정확히 확인 (거리 계산)
                        cell_center_x = (fov_cell_x + 0.5) * self.spatial_grid.cell_size
                        cell_center_y = (fov_cell_y + 0.5) * self.spatial_grid.cell_size
                        distance_sq = (cell_center_x - obj.pos.x) ** 2 + (cell_center_y - obj.pos.y) ** 2

                        if distance_sq <= fov_radius ** 2:  # 정확한 원형 FOV 판정
                            fov_cells.add((fov_cell_x, fov_cell_y))

        # 4️⃣ 활성화된 Grid (객체가 있는 셀)
        for (cell_x, cell_y) in self.spatial_grid.grid.keys():
            color = "gray"  # 기본적으로 활성화된 그리드 셀 (객체 포함)
            if (cell_x, cell_y) in fov_cells:
                color = "blue"  # FOV 영역 내의 셀을 파란색으로 강조

            rect = patches.Rectangle(
                (cell_x * self.spatial_grid.cell_size, cell_y * self.spatial_grid.cell_size),
                self.spatial_grid.cell_size, self.spatial_grid.cell_size,
                linewidth=1, edgecolor=color, facecolor='none', alpha=0.5
            )
            self.ax.add_patch(rect)
            self.grid_patches.append(rect)
