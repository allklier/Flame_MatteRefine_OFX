# ofx/CLAUDE.md — the plugin layer of Matte Refine

Inputs, colour, views, render rules, platforms, Flame host behaviour, layout, logging and
support. The operations are in `src/ops/CLAUDE.md`; goal, architecture, the decision index and
the cross-cutting decisions in the root `CLAUDE.md`. Tags as there (DECIDED / PROPOSED / OPEN /
UNVERIFIED). Keep this file under ~250 lines.

## Decisions

### Matte in, matte out

- **4. DECIDED (user, 2026-10-22): picture on the front, matte on the source alpha input.** The
  matte is always the source clip's alpha; Matte From (Alpha / Luminance / Red, 2026-10-21) is
  removed, so the RGB is free for ops that need the picture (Snap to Image, Foreground Colour).
  **Verified** (probe build, Flame 2027 on akflame5, 2026-10-22): a filter's source shows in
  Flame as a front socket plus a "source alpha" socket; a matte there arrives as the source's
  alpha (RGBA, RGB = the front). With nothing on it, alpha is a constant 1.0, so the footer
  asks for the matte when the input matte is constant. An extra optional OFX clip (mask, General
  context) appeared as a third socket and received the matte too: redundant, removed. Output
  alpha = the refined matte; output RGB per View (11). Flame forces RGBA output; alpha is the
  node's matte output (LTX decision 48). Saved nodes that used Luminance now read alpha. Again the only route since 2026-10-24 (decision 22 superseded): a filter, Source mandatory, no Matte clip.
- **22. SUPERSEDED (user, 2026-10-24) by decision 4's layout for all three plugins.** 2026-10-22 (Flame test of 0.2.0) gave the
  matte its own mask clip ("Matte", an extra socket) and made the picture optional, because with the matte only on Source's
  alpha an unconnected front left the node without size or colour information and it failed. Decision 35 (2026-10-23) and the
  Matte Merge probe (2026-10-24) then showed that connecting an extra mask clip makes Flame deliver the picture already
  converted and tag Source "Matte". Now: **the picture on the front is mandatory** (it settles size, pixel aspect and
  colourspace for Flame; a matte-only job connects the matte to both sockets), **the matte is Source's alpha**, there is no
  Matte clip, and Snap to Image / Edge Colour Unmix always have their picture. Param ids and choice indexes are unchanged,
  so saved setups load; old batches need their Matte input moved to the source alpha.
- **23. DECIDED (user, 2026-10-23, Flame test): output colourspace tags; View on both tabs.**
  Without colour management Flame tagged the output with the working colourspace (ACEScg), so
  a matte on RGB was displayed through the ACES view and looked grey. Rules (user): picture
  passing through → the input's colourspace; matte on RGB → the matte colourspace (Flame's
  configs: "Matte", alias adsk:matte_data, family Data, encoding data; 2026.0 aces2.0 config).
  Implementation: OpenFX 1.5 colour management (Flame 2026.1+ supports it, Autodesk "What's
  New 2026.1"); `ofxColour.h` + the native config header vendored from OFX_Release_1.5.1 (rest
  stays 1.4). The plugin declares OCIO style; `GetOutputColourspace` (unknown to the 1.4
  Support library, so the factory's entry point answers it and passes the rest on) returns
  `OfxColourspace_Source` for View Picture, else "Matte" if the active OCIO
  config defines it (text check of the config file), else "Raw" (Core/Full) / "ofx_raw"
  (Basic). No input conversion is
  requested, so Foreground Colour unmixes in the picture's own colourspace (correct only if
  that is scene-linear: OPEN). **View on both tabs:** a second View (tab 2's first column)
  kept equal through changedParam (user edits only; the echo is ignored), both not animatable;
  render reads tab 1's. UNVERIFIED in Flame: that Flame re-asks the output colourspace when
  View changes, and that the other tab's widget shows the new value.
- **24. DECIDED (user, 2026-10-23): Foreground Colour errors on a non-linear picture.** The
  unmixing is only correct in linear light and a log or video picture is easy to miss. While it
  is on, each render reads the front input's colourspace (kOfxImageClipPropColourspace) and
  looks up its `encoding:` in the active OCIO config text (`ocio_encoding`: name, alias or
  role; no OCIO library); anything but scene-linear, an unknown name, or no colour management
  → a footer error (the render still runs). Verified on Flame's 2026.0 aces2.0 config: ACEScg
  scene-linear, ACEScct log, Matte / Raw data. A first parser attached the named transforms'
  `encoding: log` to the last colourspace (Raw): test case added. Converting the picture to
  linear automatically (preferred colourspace) is the alternative, not built.
- **25. DECIDED (user, 2026-10-23): "Foreground Colour" is shown as "Edge Colour Unmix".** The
  old name read like another matte operation and isn't an artist's term; "Decontaminate" could
  also be read as a matte operation; it isn't despill either (nothing is replaced with the
  screen colour's opposite). Labels and messages only: the code (`foreground_colour`), the
  param names (`fg*`, `grpForegroundColour`) and saved setups are unchanged.
- **5. Render is synchronous and per frame** (14). Ops run in `render()` on Flame's main thread
  (all OFX actions do), so the algorithmic chain must stay well under ~100 ms at UHD. No
  temporal processing. Input outside 0–1 (float) is clamped to 0–1 before the ops.
- **6. Pixel-sized params scale with the OFX render scale** (proxies match); percentages don't.

- **11. DECIDED (user, 2026-10-21): View** choice for the RGB output; alpha is always the result.
  **Matte** (default since Flame test 3: RGB = the refined matte, so both outputs agree), **Show
  Filled** (RGB = the input matte in grey, added fill (out − in) tinted magenta (1, 0, 1)),
  **Source** (RGB pass-through, for a front + matte chain). Why: with Matte From = Luminance the
  front input *is* the matte, so a pass-through RGB showed the unrefined matte next to the
  refined alpha (user confused in test 3). Indexes: 0 Source (was "Result" = source), 1 Show
  Filled, 2 Matte, so saved nodes keep their output. Render-only.

### Platforms

- **13. DECIDED (user, 2026-10-21): Rocky 9 today, macOS next.** The ops stay portable (no
  Linux-only calls). Built 2026-10-22 (0.1.2): `scripts/build_mac.sh` (as LTX's), bundle
  `Contents/MacOS/MatteRefine.ofx` (arm64) + generated `Info.plist`, exports via the vendored
  `osxSymbols`, links only libc++/libSystem; `CMAKE_OSX_DEPLOYMENT_TARGET` 15.0 (else the build
  Mac's version: minos 26); ctest 3/3 on an M1 Ultra. Install dir `/Library/OFX/Plugins`.
  NOT yet tried in Flame on a Mac; no release package script for it yet.

## Flame host behaviour (from LTX `ofx/CLAUDE.md`, verified there on Flame 2027.1/2027.2)

- Every OFX action, render included, runs on Flame's **main UI thread**. Flame renders the current
  frame on Batch load and on every frame change, revisit and param change. Frames are **32-bit
  float RGBA**; OFX time starts at 0.
- **Every plugin `setValue` comes straight back as `changedParam`**: reacting recursed and
  crashed Flame. Handle only `eChangeUserEdit`, use a re-entrancy guard, write labels only when
  they change.
- **Layout:** params flow into columns of 5 rows, up to 6 columns per tab, in definition order;
  each group starts a column; pages are ignored (a new tab only on overflow, named after the next
  OFX page); rows: choice 2, boolean 1, double 1, button 1, plain string 1 (no label), label
  string 1; text cut at ~30 characters. Don't use `setPageParamOrder` (throws). Int rows: UNVERIFIED.
  **A secret (hidden) param still takes a slot** (Flame test, 2026-10-23): the node ID defined
  last outside any group made its own empty column and, once tab 2 was full, a third tab. Keep
  hidden params inside a group that has a free row (now Support).
- `getClipPreferences` → Alpha-only is ignored: output stays RGBA (image + matte outputs).
- Persistent messages show in the footer; `sendMessage` shows popups. Flame may not show param
  hints, so labels say what they do.
- **Doubles without `setDigits`/`setIncrement` showed as integers** (Flame test 1, 2026-10-21:
  Solid Above couldn't be set between 0 and 1). Now digits 3/2/1, increment 0.01/0.1/0.5
  (`ofxsParams.cpp:494-503`); whether Flame honours them: UNVERIFIED.

**Planned layout (tab 1):** Matte (View 2, Version label 1) · Fill Holes (Enable,
Threshold, Max Hole %, Bridge Gaps, Rim) · Fill Holes Edges (4 booleans) · Solid Core (Enable,
Depth px, Feather px, Edge At) · Support (Debug Logging, Collect Logs, Save Matte). Five columns; one left
before tab 2.

## Logging and support (LTX decision 39, scaled down)

- One log `~/.cache/matte-refine/logs/matte-refine.log` (rotated at plugin load, 10 MB × 5),
  LTX's line format (`YYYY-MM-DD HH:MM:SS.mmm LEVEL comp[pid/tid] node=<id> | msg`); environment
  snapshot at load (version + git, host, OS, glibc, CPU); INFO for user actions and param edits;
  DEBUG for renders with per-op timings. Never logged: image data, clip or project names.
- **Collect Logs** button → `~/matte-refine-support-<ts>.tar.gz`.
- **Save Matte** button → this frame's input matte as raw float32 (top row first) + a `.txt`
  header in `~/.cache/matte-refine/dumps/`, for exact analysis of a failing frame. Footage can
  be confidential: dumps stay on the machine; analyse them there or in a private scratch
  folder, never upload or commit them.
- DEBUG render line: settings, source channels, input matte min/mean/max, holes filled / too
  large / notches kept, % below threshold, % outside, per-op time.
- Version label on the node (version + git hash) so a Flame test shows which build is loaded.
