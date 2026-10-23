// colourspaceEncoding (decision 24) on a small config in the layout of Flame's OCIO v2 configs.
#include <cstdio>
#include <string>

#include "ocio_encoding.h"

namespace {

int failures = 0;
#define CHECK(cond)                                               \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

const char* kConfig = R"(ocio_profile_version: 2.4

roles:
  compositing_log: ACEScct
  scene_linear: ACEScg

display_colorspaces:
  - !<ColorSpace>
    name: sRGB - Display
    aliases: [sRGB display, srgb_display]
    encoding: sdr-video

colorspaces:
  - !<ColorSpace>
    name: ACEScg
    aliases: [acescg_ap1, "ocio:acescg_ap1_scene"]
    family: ACES
    encoding: scene-linear
    allocation: uniform

  - !<ColorSpace>
    name: ACEScct
    aliases: [acescct_ap1]
    to_scene_reference: !<GroupTransform>
      children:
        - !<MatrixTransform> {matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]}
        - !<ExponentTransform> {value: 2.2}
    encoding: log

  - !<ColorSpace>
    name: Matte
    aliases: [adsk:matte_data]
    encoding: data

  - !<ColorSpace>
    name: Old Style
    family: Legacy

  - !<ColorSpace>
    name: Raw
    aliases: [none, data]
    encoding: data

named_transforms:
  - !<NamedTransform>
    name: ACEScct - Curve
    encoding: log
)";

}  // namespace

int main() {
  using mr::colourspaceEncoding;
  CHECK(colourspaceEncoding(kConfig, "ACEScg") == "scene-linear");
  CHECK(colourspaceEncoding(kConfig, "acescg_ap1") == "scene-linear");       // alias
  CHECK(colourspaceEncoding(kConfig, "ocio:acescg_ap1_scene") == "scene-linear");  // quoted alias
  CHECK(colourspaceEncoding(kConfig, "scene_linear") == "scene-linear");     // role
  CHECK(colourspaceEncoding(kConfig, "compositing_log") == "log");
  CHECK(colourspaceEncoding(kConfig, "ACEScct") == "log");
  CHECK(colourspaceEncoding(kConfig, "sRGB - Display") == "sdr-video");      // display space
  CHECK(colourspaceEncoding(kConfig, "adsk:matte_data") == "data");
  CHECK(colourspaceEncoding(kConfig, "Old Style").empty());                  // no encoding
  CHECK(colourspaceEncoding(kConfig, "Nope").empty());
  // Later items (named transforms) have encodings too: they must not land on the last space.
  CHECK(colourspaceEncoding(kConfig, "Raw") == "data");
  CHECK(colourspaceEncoding(kConfig, "ACEScct - Curve").empty());
  CHECK(colourspaceEncoding("", "ACEScg").empty());
  std::printf("%s (%d failure(s))\n", failures ? "FAILED" : "OK", failures);
  return failures ? 1 : 0;
}
