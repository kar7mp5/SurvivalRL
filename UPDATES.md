# SurvivalRL Updates

-   [**Circle and Rectangle Collision System**](#circle-and-rectangle-collision-system)
-   [**Plant, Herbivore, and Predator**](#plant-herbivore-and-predator)
-   [**Multi Simulation with Population Graph**](#multi-simulation-with-population-graph)
-   [**Debug Mode**](#debug-mode)
-   [**POV Detector**](#pov-detector)
-   [**Spatial Hash Grid (Collision)**](#spatial-hash-grid-collision)

## Circle and Rectangle Collision System

**Date: 2025.03.03.** - [**Commit Link**](https://github.com/kar7mp5/SurvivalRL/commit/27968305b239fde20802f6942b82fc8bffc1c955)

**Added Functions:**

-   Add Game Object
-   Added Objects
    -   Circle obj
    -   Rectangle obj

![circle_and_rect_collision](./docs/circle_and_rect_collision.gif)

## Plant, Herbivore, and Predator

**Date: 2025.03.03** - [**Commit Link**](https://github.com/kar7mp5/SurvivalRL/commit/f1abcdb2f6f2a5673954d47ea583f328184bf76e)

**Improvement:**

-   Improve OOC
-   Improve rectangle collision algorithm (AABB -> GJK)
-   Add Plant, Herbivore, and Predator (but these objects have no functions).

![plant_herbivore_predator_v1](./docs/plant_herbivore_predator_v1.gif)
![plant_herbivore_predator_v1_1](./docs/plant_herbivore_predator_v1-1.gif)

## Multi Simulation with Population Graph

**Date: 2025.03.03** - [**Commit Link**](https://github.com/kar7mp5/SurvivalRL/commit/a57ec7a8435e497bd419ead2e381c5bc6b5a60f0)

-   Add multi simulations and plots.
-   Duplication test.

![multi_simulation_with_population_graphs](./docs/multi_simulation_with_population_graphs.gif)

## Debug Mode

**Date: 2025.03.03** - [**Commit Link**](https://github.com/kar7mp5/SurvivalRL/commit/e69b6ca8c2296b0b65660438d7e64ca53ecedcfa)

-   Add energy but it has no functions.
-   Add Debug Mode.

![single_simulation_with_population_graph_debug](./docs/single_simulation_with_population_graph_debug.gif)

## FOV Detector

**Date: 2025.03.05.** - [**Commit Link**](https://github.com/kar7mp5/SurvivalRL/commit/03d710c4adadf3cd7c79e6b35f0317f1845243c7)

-   Add individual debug mode.
-   Add FOV detector.

![POV_detection](./docs/POV_detection.gif)

## Spatial Hash Grid (Collision)

**Data: 2025.03.13.** - [**Commit Link**](https://github.com/kar7mp5/SurvivalRL/commit/db75f806d56dd9e86ce1deced4907db306a688ee)

-   At first, I tried using a QuadTree for object collision detection. It worked well for collisions, but I stopped using it due to issues with FOV detection. So, I switched to a Spatial Hash Grid instead.
-   I set the predator object's size to change based on its energy level.

![spatial_hash_grid](./docs/spatial_hash_grid.gif)
