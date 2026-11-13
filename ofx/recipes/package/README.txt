Edge Recipes for Autodesk Flame (Linux, Rocky Linux 9)

Edge Recipes is one OFX plugin with recipe nodes under the "Edge Recipes" group in Flame. The first
recipe is Hair Locks. This package holds only EdgeRecipes.ofx.bundle, which already contains every
method it uses (its own model runner and ONNX Runtime included). Matte Refine, Matte Refine ML and
Matte Merge are separate packages: install them as well if you want those nodes; all can be installed
side by side, and each package installs and uninstalls only its own bundle.

Install (as an admin, with Flame closed):
  sudo ./install.sh                 copies EdgeRecipes.ofx.bundle to /usr/OFX/Plugins
  sudo ./install.sh --uninstall     removes it again (other plugins stay)

Models (each artist, as their normal user, once):
  ./setup.sh                        licences, then the models and NVIDIA CUDA libraries (about 1.9 GB)
                                    into ~/.local/share/matte-refine-ml
The steps that use a model (Edge Model in Hair Locks) need setup.sh and an NVIDIA GPU; every other step
works without it. setup.sh is the same as Matte Refine ML's: if you already ran it for Matte Refine ML,
nothing more is needed (one licence acceptance, one model folder for both).
  ./setup.sh --remove               deletes the models (it warns when Matte Refine ML still uses them)

Then start Flame. USERGUIDE.md (section "Edge Recipes") explains the node; licences.txt and THIRD_PARTY/
hold the licences.
