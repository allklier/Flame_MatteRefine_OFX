# Plate Tools (docs/plate_merge.md, root decision 43): only with -DMR_BUILD_PLATE=ON (needs MR_BUILD_MERGE
# for its distance transform and the systest host). Classical, no model: the core (pure C++14, no OFX),
# its synthetic-scene tests, then the plugin bundle PlateTools.ofx.bundle (first effect: Plate Merge).
add_library(mr_plate_core STATIC src/plate_merge.cpp)
target_include_directories(mr_plate_core PUBLIC src)
target_link_libraries(mr_plate_core PUBLIC mr_merge_core mr_ops)
target_compile_options(mr_plate_core PRIVATE -Wall -Wextra -fvisibility=hidden)

add_executable(test_plate_merge tests/test_plate_merge.cpp)
target_link_libraries(test_plate_merge PRIVATE mr_plate_core)
add_test(NAME plate_core COMMAND test_plate_merge smoke)
set_tests_properties(plate_core PROPERTIES LABELS smoke)
add_test(NAME plate_scenes COMMAND test_plate_merge scenes)

# The plugin (as Matte Merge: C++14, only the two OFX entry points, libstdc++ dynamic): bundle
# PlateTools.ofx.bundle, its own version; one effect per factory (Plate Merge first, root decision 43).
set(PT_VERSION "0.1.0")
string(REPLACE "." ";" PT_VERSION_PARTS "${PT_VERSION}")
list(GET PT_VERSION_PARTS 0 PT_VERSION_MAJOR)
list(GET PT_VERSION_PARTS 1 PT_VERSION_MINOR)
set(PT_BUNDLE_DIR ${CMAKE_BINARY_DIR}/PlateTools.ofx.bundle)

# Plugin logic without OFX types (param mapping, footers, colourspace answer): unit-tested.
add_library(mr_plate_ui STATIC src/plate_ui.cpp)
target_include_directories(mr_plate_ui PUBLIC src PRIVATE ${OFX_DIR}/include)
target_link_libraries(mr_plate_ui PUBLIC mr_plate_core)
target_compile_options(mr_plate_ui PRIVATE -Wall -Wextra -fvisibility=hidden)

add_library(PlateTools MODULE src/plugin_plate.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/log.cpp ${CMAKE_SOURCE_DIR}/ofx/src/user_config.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/process.cpp ${CMAKE_SOURCE_DIR}/ofx/src/env_info.cpp
  ${CMAKE_SOURCE_DIR}/ofx/src/ocio_encoding.cpp ${CMAKE_SOURCE_DIR}/${MR_PLATFORM_SOURCES})
target_link_libraries(PlateTools PRIVATE ofx_support mr_plate_ui mr_plate_core mr_merge_core mr_ops Threads::Threads)
target_include_directories(PlateTools PRIVATE ${CMAKE_SOURCE_DIR}/ofx/src ${OFX_DIR}/Support/Library)
# MR_VERSION: env_info's "plugin: Matte Refine <MR_VERSION>" line reads "Matte Refine Plate Tools 0.1.0".
target_compile_definitions(PlateTools PRIVATE PT_VERSION="${PT_VERSION}" MR_GIT="${MR_GIT}"
  MR_VERSION="Plate Tools ${PT_VERSION}" PT_VERSION_MAJOR=${PT_VERSION_MAJOR} PT_VERSION_MINOR=${PT_VERSION_MINOR})
target_compile_options(PlateTools PRIVATE -Wall -Wextra -Wno-deprecated-declarations -Wno-deprecated
  -fvisibility=hidden)
if(APPLE)  # decision 13: as Matte Refine (root CMakeLists.txt)
  target_link_options(PlateTools PRIVATE -Wl,-exported_symbols_list,${OFX_DIR}/Support/include/osxSymbols)
  file(WRITE ${PT_BUNDLE_DIR}/Contents/Info.plist
"<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
  <key>CFBundleExecutable</key><string>PlateTools.ofx</string>
  <key>CFBundleIdentifier</key><string>com.janklier.platetools</string>
  <key>CFBundleName</key><string>Plate Tools</string>
  <key>CFBundlePackageType</key><string>BNDL</string>
  <key>CFBundleVersion</key><string>${PT_VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${PT_VERSION}</string>
</dict>
</plist>
")
else()
  target_link_options(PlateTools PRIVATE
    -Wl,--version-script=${OFX_DIR}/Support/include/linuxSymbols -Wl,--no-undefined)
endif()
set_target_properties(PlateTools PROPERTIES PREFIX "" SUFFIX ".ofx"
  LIBRARY_OUTPUT_DIRECTORY ${PT_BUNDLE_DIR}/Contents/${MR_ARCH_DIR})

add_executable(test_plate_ui tests/test_plate_ui.cpp)
target_include_directories(test_plate_ui PRIVATE ${OFX_DIR}/include)
target_link_libraries(test_plate_ui PRIVATE mr_plate_ui)
add_test(NAME plate_ui COMMAND test_plate_ui)
set_tests_properties(plate_ui PROPERTIES LABELS smoke)
add_executable(test_load_bundle_plate tests/test_load_bundle_plate.cpp)
target_include_directories(test_load_bundle_plate PRIVATE ${OFX_DIR}/include)
target_link_libraries(test_load_bundle_plate PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(test_load_bundle_plate PlateTools)
add_test(NAME plate_load_bundle COMMAND test_load_bundle_plate ${PT_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/PlateTools.ofx)
set_tests_properties(plate_load_bundle PROPERTIES LABELS smoke)
if(NOT APPLE)  # the generic exports check of Matte Merge (takes the binary as its argument)
  add_test(NAME plate_bundle_exports COMMAND sh ${CMAKE_SOURCE_DIR}/ofx/merge/tests/check_exports.sh
    ${PT_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/PlateTools.ofx)
endif()

# System test: PlateTools.ofx in Matte Merge's Flame-like host; the render equals mergePlates() called directly.
add_executable(plate_systest systest/plate_systest.cpp ${CMAKE_SOURCE_DIR}/ofx/merge/systest/ofx_host.cpp)
target_include_directories(plate_systest PRIVATE ${CMAKE_SOURCE_DIR}/ofx/merge/systest ${OFX_DIR}/include)
target_link_libraries(plate_systest PRIVATE mr_plate_ui ${CMAKE_DL_LIBS} Threads::Threads)
add_dependencies(plate_systest PlateTools)
add_test(NAME plate_systest COMMAND plate_systest ${PT_BUNDLE_DIR}/Contents/${MR_ARCH_DIR}/PlateTools.ofx
  ${CMAKE_CURRENT_BINARY_DIR}/systest_home)
set_tests_properties(plate_systest PROPERTIES TIMEOUT 300)
