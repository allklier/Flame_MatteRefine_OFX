# Edge Recipes (decision 41, docs/edge_recipes.md). Each work package owns one sub-folder with its own
# CMakeLists.txt; a folder that does not exist yet is skipped. The Flame probe (ofx/recipes/probe) has its own option.
#   core    step interface, evidence cache, planner         (WP2)
#   tracer  strand tracer, library mr_tracer                (WP3)
#   model   model candidate, library mr_recipe_model        (WP4)
#   vote    strand presence vote over neighbour frames   (WP6)
#   plugin  the EdgeRecipes bundle, one effect per recipe   (WP1, WP7)
foreach(sub core tracer model vote plugin scoring)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${sub}/CMakeLists.txt)
    add_subdirectory(${sub})
  endif()
endforeach()
