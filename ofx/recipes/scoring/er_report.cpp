// er_score report: scores.csv + measures.csv (er_score sweep) -> summary.md. Tables: headline scores of the
// default setting, per look / width / margin / distractor, the control sensitivity, the vote x Temporal flicker
// matrix, the sequence ends, the proxy, the contract checks, and the draft Analyze rules (decision stumps on one
// cheap measurement each, fitted to the per-scene best setting, with their hit rates and a 2-fold cross-check).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Row = std::map<std::string, std::string>;

std::vector<Row> readCsv(const std::string& path) {
  std::vector<Row> rows;
  std::ifstream f(path);
  std::string line;
  std::vector<std::string> head;
  auto split = [](const std::string& s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ',')) v.push_back(x);
    return v;
  };
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const std::vector<std::string> v = split(line);
    if (head.empty() || v[0] == "set") {  // a header (appended files repeat none, but be safe)
      head = v;
      continue;
    }
    Row r;
    for (size_t i = 0; i < head.size() && i < v.size(); ++i) r[head[i]] = v[i];
    rows.push_back(r);
  }
  return rows;
}

double num(const Row& r, const std::string& k) {
  auto it = r.find(k);
  if (it == r.end()) return NAN;
  try {
    return std::stod(it->second);
  } catch (...) {
    return NAN;
  }
}
const std::string& str(const Row& r, const std::string& k) {
  static const std::string empty;
  auto it = r.find(k);
  return it == r.end() ? empty : it->second;
}

double mean(const std::vector<double>& v) {
  double s = 0;
  int n = 0;
  for (double x : v)
    if (!std::isnan(x)) s += x, ++n;
  return n ? s / n : NAN;
}

using Pred = std::function<bool(const Row&)>;
std::vector<double> col(const std::vector<Row>& rows, const Pred& p, const std::function<double(const Row&)>& f) {
  std::vector<double> v;
  for (const Row& r : rows)
    if (p(r)) v.push_back(f(r));
  return v;
}
std::function<double(const Row&)> c(const std::string& k) {
  return [k](const Row& r) { return num(r, k); };
}
double flickEx(const Row& r) { return num(r, "flicker") - num(r, "flickerTruth"); }

std::string f3(double v) {
  if (std::isnan(v)) return "-";
  char b[32];
  std::snprintf(b, sizeof b, "%.3f", v);
  return b;
}
std::string f4(double v) {
  if (std::isnan(v)) return "-";
  char b[32];
  std::snprintf(b, sizeof b, "%.4f", v);
  return b;
}

bool isSet(const Row& r, const char* s) { return str(r, "set") == s; }
bool scale1(const Row& r) { return num(r, "scale") == 1.0; }

// The standard quality columns for a group of rows.
const char* kQualHead = "| coverage | IoU@0.5 | SAD/truth | false in region | distractor alpha | distractor hit rate | "
                        "flicker excess | n |\n|---|---|---|---|---|---|---|---|\n";
std::string qual(const std::vector<Row>& rows, const Pred& p) {
  const std::vector<double> cov = col(rows, p, c("coverage"));
  return "| " + f3(mean(cov)) + " | " + f3(mean(col(rows, p, c("iou")))) + " | " + f3(mean(col(rows, p, c("sad")))) +
         " | " + f4(mean(col(rows, p, c("falseIn")))) + " | " + f4(mean(col(rows, p, c("distrMass")))) + " | " +
         f3(mean(col(rows, p, c("distrHitRate")))) + " | " + f4(mean(col(rows, p, flickEx))) + " | " +
         std::to_string(cov.size()) + " |\n";
}

// ---- calibration ----------------------------------------------------------------------------------------------
struct Target {
  std::string name;
  std::vector<double> values;              // candidate settings
  std::map<std::string, std::set<int>> ok;  // scene -> acceptable value indexes (within tolerance of the best)
  std::map<std::string, int> best;          // scene -> best value index
};

struct Stump {
  std::string measure = "(constant)";
  double thr = 0;
  int below = 0, above = 0;
  int hits = 0, n = 0;
};

Stump fitStump(const Target& T, const std::vector<std::string>& scenes, const std::map<std::string, Row>& meas,
               const std::vector<std::string>& measures) {
  Stump best;
  const int nv = int(T.values.size());
  // constant
  for (int v = 0; v < nv; ++v) {
    int h = 0;
    for (const std::string& s : scenes) h += T.ok.at(s).count(v);
    if (h > best.hits) best.hits = h, best.below = best.above = v, best.measure = "(constant)";
  }
  best.n = int(scenes.size());
  for (const std::string& m : measures) {
    std::vector<double> xs;
    for (const std::string& s : scenes) xs.push_back(num(meas.at(s), m));
    std::vector<double> u = xs;
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    for (size_t k = 0; k + 1 < u.size(); ++k) {
      const double thr = 0.5 * (u[k] + u[k + 1]);
      for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b) {
          int h = 0;
          for (size_t i = 0; i < scenes.size(); ++i) h += T.ok.at(scenes[i]).count(xs[i] < thr ? a : b);
          if (h > best.hits) best = Stump{m, thr, a, b, h, int(scenes.size())};
        }
    }
  }
  return best;
}
int evalStump(const Stump& st, const Target& T, const std::vector<std::string>& scenes,
              const std::map<std::string, Row>& meas) {
  int h = 0;
  for (const std::string& s : scenes) {
    const int v = st.measure == "(constant)" ? st.below : num(meas.at(s), st.measure) < st.thr ? st.below : st.above;
    h += T.ok.at(s).count(v);
  }
  return h;
}

}  // namespace

int reportMain(const std::string& out) {
  const std::vector<Row> rows = readCsv(out + "/scores.csv");
  const std::vector<Row> mrows = readCsv(out + "/measures.csv");
  if (rows.empty()) {
    std::fprintf(stderr, "no rows in %s/scores.csv\n", out.c_str());
    return 1;
  }
  std::map<std::string, Row> meas;
  for (const Row& r : mrows) meas[str(r, "scene")] = r;
  std::ofstream md(out + "/summary.md");
  std::set<std::string> sceneSet;
  for (const Row& r : rows) sceneSet.insert(str(r, "scene"));
  md << "# Hair Locks end-to-end scores (er_score, Edge Recipes WP8)\n\n"
     << "Synthetic sequences only (er_scenes.h; `wp6` = the vote system test's sequence), driven through the Flame-like "
        "host. CLASSICAL PATH ONLY: Edge Model is off in every row (no model weights on the scoring machine), so the "
        "model candidate, Strand Gain and Edge Model are NOT scored. Picture Colour = Bypass (linear scenes). "
     << sceneSet.size() << " scenes, " << rows.size() << " rows.\n\n"
     << "Metrics (inside the drawn region, mask > 0; target = base + (1 - base) strand coverage): coverage = added alpha "
        "that is truth / missing truth; IoU at 0.5 on pixels the base had below 0.5; SAD = |out - target| / missing "
        "truth; false in region = alpha added where the truth adds < 0.02, / missing truth; distractor alpha = alpha "
        "added on distractor pixels / missing truth; hit rate = share of frames with more than 1.0 alpha unit on a "
        "distractor; flicker excess = motion-compensated mean |a_t - a_t-1| in the region minus the same for the "
        "target; edge error = mean |out - true head| on the head's edge band outside the region.\n\n";

  const Pred def = [](const Row& r) { return str(r, "setting") == "default" && scale1(r); };
  // 1. headline
  md << "## 1. Default setting (Rebuild on, Min Confidence 0.45, Feather 0.25 %, Loose on, Vote on, Snap 1, Temporal "
        "1), per scene set\n\n| set " << kQualHead;
  for (const char* s : {"main", "distractor", "detached", "vote", "wp6"})
    md << "| " << s << " " << qual(rows, [&](const Row& r) { return def(r) && isSet(r, s); });
  md << "\n### Per look (main + distractor sets)\n\n| look " << kQualHead;
  for (const char* l : {"dark_skin", "dark_texture", "dark_shadow", "light_skin"})
    md << "| " << l << " "
       << qual(rows, [&](const Row& r) { return def(r) && (isSet(r, "main") || isSet(r, "distractor")) && str(r, "look") == l; });
  md << "\n### Per strand width (main set)\n\n| width px " << kQualHead;
  for (const char* w : {"1.5", "2.5", "4.0", "6.0"})
    md << "| " << w << " " << qual(rows, [&](const Row& r) { return def(r) && isSet(r, "main") && str(r, "width") == w; });
  md << "\n### Per region looseness (main set)\n\n| margin px " << kQualHead;
  for (const char* m : {"0", "10", "20"})
    md << "| " << m << " " << qual(rows, [&](const Row& r) { return def(r) && isSet(r, "main") && str(r, "margin") == m; });
  md << "\n### Per look x width, coverage / false in region (main set, default)\n\n| look | 1.5 | 2.5 | 4.0 | 6.0 |\n|---|---|---|---|---|\n";
  for (const char* l : {"dark_skin", "dark_texture", "dark_shadow", "light_skin"}) {
    md << "| " << l;
    for (const char* w : {"1.5", "2.5", "4.0", "6.0"}) {
      const Pred p = [&](const Row& r) { return def(r) && isSet(r, "main") && str(r, "look") == l && str(r, "width") == w; };
      md << " | " << f3(mean(col(rows, p, c("coverage")))) << " / " << f3(mean(col(rows, p, c("falseIn"))));
    }
    md << " |\n";
  }

  // 2. distractors
  md << "\n## 2. Distractors (distractor set: width 3, region 10 px loose, every look; default and variants)\n\n"
     << "| distractor | setting | distractor alpha | hit rate | coverage | false in region | crossing strand coverage |\n"
     << "|---|---|---|---|---|---|---|\n";
  for (const char* d : {"glove", "glasses", "mole", "crossing", "tipband"})
    for (const char* st : {"default", "snap_off", "loose_off", "vote_off", "mc0.70_fe0.25"}) {
      const Pred p = [&](const Row& r) { return scale1(r) && isSet(r, "distractor") && str(r, "distractor") == d && str(r, "setting") == st; };
      md << "| " << d << " | " << st << " | " << f4(mean(col(rows, p, c("distrMass")))) << " | "
         << f3(mean(col(rows, p, c("distrHitRate")))) << " | " << f3(mean(col(rows, p, c("coverage")))) << " | "
         << f4(mean(col(rows, p, c("falseIn")))) << " | " << f3(mean(col(rows, p, c("crossCov")))) << " |\n";
    }
  md << "\nDistractor hit rate per look (distractor set, default):\n\n| distractor | dark_skin | dark_texture | dark_shadow | light_skin |\n|---|---|---|---|---|\n";
  for (const char* d : {"glove", "glasses", "mole", "tipband"}) {
    md << "| " << d;
    for (const char* l : {"dark_skin", "dark_texture", "dark_shadow", "light_skin"}) {
      const Pred p = [&](const Row& r) { return def(r) && isSet(r, "distractor") && str(r, "distractor") == d && str(r, "look") == l; };
      md << " | " << f3(mean(col(rows, p, c("distrHitRate"))));
    }
    md << " |\n";
  }

  md << "\n### Detached lock 0 (starts 18 px off the head: unanchored), per Min Confidence and Loose Pieces\n\n"
     << "| setting | coverage | SAD | false in region | distractor alpha | distractor hit rate |\n|---|---|---|---|---|---|\n";
  for (const char* st : {"mc0.25_fe0.25", "mc0.35_fe0.25", "default", "mc0.55_fe0.25", "mc0.70_fe0.25", "loose_off"}) {
    const Pred p = [&](const Row& r) { return scale1(r) && isSet(r, "detached") && str(r, "setting") == st && str(r, "look") != "light_skin"; };
    md << "| " << st << " | " << f3(mean(col(rows, p, c("coverage")))) << " | " << f3(mean(col(rows, p, c("sad")))) << " | "
       << f4(mean(col(rows, p, c("falseIn")))) << " | " << f4(mean(col(rows, p, c("distrMass")))) << " | "
       << f3(mean(col(rows, p, c("distrHitRate")))) << " |\n";
  }
  md << "\n(light_skin excluded here: the tracer finds no pieces on that look.)\n";

  // 3. sensitivity
  const Pred ms = [](const Row& r) { return scale1(r) && (isSet(r, "main") || isSet(r, "distractor") || isSet(r, "detached")); };
  md << "\n## 3. Sensitivity (main + distractor + detached sets; one control moved from the default)\n\n"
     << "| control | value | coverage | SAD | false in region | distractor alpha | flicker excess | edge error outside | "
        "identical to default |\n|---|---|---|---|---|---|---|---|---|\n";
  std::map<std::string, std::string> defHash;
  for (const Row& r : rows)
    if (def(r)) defHash[str(r, "scene")] = str(r, "hash");
  auto sens = [&](const std::string& ctrl, const std::string& label, const std::string& setting) {
    const Pred p = [&](const Row& r) { return ms(r) && str(r, "setting") == setting; };
    int same = 0, n = 0;
    for (const Row& r : rows)
      if (p(r)) ++n, same += defHash[str(r, "scene")] == str(r, "hash");
    md << "| " << ctrl << " | " << label << " | " << f3(mean(col(rows, p, c("coverage")))) << " | "
       << f3(mean(col(rows, p, c("sad")))) << " | " << f4(mean(col(rows, p, c("falseIn")))) << " | "
       << f4(mean(col(rows, p, c("distrMass")))) << " | " << f4(mean(col(rows, p, flickEx))) << " | "
       << f4(mean(col(rows, p, c("edgeErr")))) << " | " << same << "/" << n << " |\n";
  };
  for (const char* mc : {"0.25", "0.35", "0.45", "0.55", "0.70"})
    sens("Min Confidence", mc, std::string(mc) == "0.45" ? "default" : std::string("mc") + mc + "_fe0.25");
  for (const char* fe : {"0.0", "0.25", "0.5", "1.0", "2.0"})
    sens("Region Feather %", fe, std::string(fe) == "0.25" ? "default" : std::string("mc0.45_fe") + fe);
  sens("Strand Gain", "0.8", "gain0.8");
  sens("Strand Gain", "1.0", "default");
  sens("Strand Gain", "1.5", "gain1.5");
  sens("Loose Pieces", "on", "default");
  sens("Loose Pieces", "off", "loose_off");
  sens("Temporal Amount", "0", "tm0.0");
  sens("Temporal Amount", "0.5", "tm0.5");
  sens("Temporal Amount", "1", "default");
  sens("Temporal", "off", "tm_off");
  sens("Snap Amount", "0", "snap0.0");
  sens("Snap Amount", "0.5", "snap0.5");
  sens("Snap Amount", "1", "default");
  sens("Snap to Picture", "off", "snap_off");
  sens("Temporal Vote", "on", "default");
  sens("Temporal Vote", "off", "vote_off");
  sens("Rebuild Detail", "off", "rebuild_off");
  sens("Snap only (Rebuild, Temporal off)", "-", "snap_only");

  // 4. flicker matrix + vote
  md << "\n## 4. Temporal Vote x Temporal (flicker of the strand alpha; ends)\n\n"
     << "| set | vote | Temporal | coverage | flicker | truth flicker | coverage std over frames | coverage end frames | "
        "coverage middle frames | dip-frame coverage | spurious stroke alpha | edge error outside |\n"
     << "|---|---|---|---|---|---|---|---|---|---|---|---|\n";
  for (const char* s : {"main", "distractor", "detached", "vote", "wp6"})
    for (const char* st : {"default", "vote_off", "tm_off", "vote_off_tm_off"}) {
      const Pred p = [&](const Row& r) { return scale1(r) && isSet(r, s) && str(r, "setting") == st; };
      const bool vote = std::string(st).find("vote_off") == std::string::npos;
      const bool tm = std::string(st).find("tm_off") == std::string::npos;
      md << "| " << s << " | " << (vote ? "on" : "off") << " | " << (tm ? "on" : "off") << " | "
         << f3(mean(col(rows, p, c("coverage")))) << " | " << f4(mean(col(rows, p, c("flicker")))) << " | "
         << f4(mean(col(rows, p, c("flickerTruth")))) << " | " << f3(mean(col(rows, p, c("covStd")))) << " | "
         << f3(mean(col(rows, p, c("covEnds")))) << " | " << f3(mean(col(rows, p, c("covMid")))) << " | "
         << f3(mean(col(rows, p, c("dipCov")))) << " | " << f3(mean(col(rows, p, c("spurMass")))) << " | "
         << f4(mean(col(rows, p, c("edgeErr")))) << " |\n";
    }

  // 5. proxy
  md << "\n## 5. Proxy (render scale 0.5: the scene box-downsampled 2x) vs full scale, default setting\n\n"
     << "| set | width px (full res) | coverage 1.0 | coverage 0.5 | SAD 1.0 | SAD 0.5 | false in region 1.0 | 0.5 |\n"
     << "|---|---|---|---|---|---|---|---|\n";
  for (const char* s : {"main", "distractor", "detached"})
    for (const char* w : {"1.5", "2.5", "3.0", "4.0", "6.0"}) {
      auto p = [&](double sc) {
        return Pred([&, sc](const Row& r) { return num(r, "scale") == sc && isSet(r, s) && str(r, "width") == w && str(r, "setting") == "default"; });
      };
      if (col(rows, p(1.0), c("coverage")).empty()) continue;
      md << "| " << s << " | " << w << " | " << f3(mean(col(rows, p(1.0), c("coverage")))) << " | "
         << f3(mean(col(rows, p(0.5), c("coverage")))) << " | " << f3(mean(col(rows, p(1.0), c("sad")))) << " | "
         << f3(mean(col(rows, p(0.5), c("sad")))) << " | " << f4(mean(col(rows, p(1.0), c("falseIn")))) << " | "
         << f4(mean(col(rows, p(0.5), c("falseIn")))) << " |\n";
    }

  // 6. contracts
  long long belowFull = 0, outsideTmOff = 0, belowMaskTmOff = 0, outsideTruth0TmOff = 0;
  int tmOffRows = 0;
  std::set<std::string> where;
  double outsideMaxTmOn = 0;
  for (const Row& r : rows) {
    belowFull += (long long)num(r, "belowFull");
    if (num(r, "belowFull") > 0) where.insert(str(r, "scene") + ":" + str(r, "setting"));
    if (num(r, "temporal") == 0) {
      ++tmOffRows;
      outsideTmOff += (long long)num(r, "outsideChanged");
      belowMaskTmOff += (long long)num(r, "belowMask");
      outsideTruth0TmOff += (long long)num(r, "outsideAddedTruth0");
      if (num(r, "outsideChanged") > 0 || num(r, "belowMask") > 0) where.insert(str(r, "scene") + ":" + str(r, "setting"));
    } else {
      outsideMaxTmOn = std::max(outsideMaxTmOn, num(r, "outsideMaxAbs"));
    }
  }
  md << "\n## 6. Contract checks (every frame of every row)\n\n"
     << "- below the base where the region weight is 1: " << belowFull << " px over all " << rows.size() << " rows\n"
     << "- rows with Temporal off (" << tmOffRows << "): changed outside the region " << outsideTmOff
     << " px; alpha added outside where the truth is 0: " << outsideTruth0TmOff << " px; below the base inside the region "
     << belowMaskTmOff << " px\n"
     << "- rows with Temporal on: Temporal changes the matte outside the region by design (largest change "
     << f3(outsideMaxTmOn) << "); see the edge error columns for whether it helps\n";
  if (!where.empty()) {
    md << "- violations in:";
    int k = 0;
    for (const std::string& w : where)
      if (k++ < 20) md << " " << w;
    md << "\n";
  }

  // 7. calibration
  md << "\n## 7. Draft Analyze rules (WP10 calibration)\n\n"
     << "Per scene (main + distractor + detached sets, scale 1) the best setting of each control Analyze would set, from the grid "
        "rows; a value is ACCEPTABLE when its score is within the tolerance of the scene's best. Inside controls "
        "(Rebuild on/off, Min Confidence at Feather 0.25, Feather at Min Confidence 0.45) are scored by SAD "
        "(tolerance max(0.02, 5 % of the best)); Temporal Amount by the edge error outside the region (tolerance 5 % "
        "of the best). A rule = one measurement of the centre frame and one threshold (a decision stump), fitted by "
        "exhaustive search; hit rate = share of scenes where the rule's value is acceptable; 2-fold = fitted on even, "
        "tested on odd scenes and vice versa. Strand Gain, Edge Model: not calibrated (model not scored). Jump Repair "
        "Min Size: not calibrated (no jump scenes here; decision 42).\n\n";
  std::vector<std::string> scenes;
  for (const Row& r : rows)
    if (def(r) && (isSet(r, "main") || isSet(r, "distractor") || isSet(r, "detached")) && meas.count(str(r, "scene"))) scenes.push_back(str(r, "scene"));
  std::map<std::string, std::map<std::string, Row>> bySet;  // scene -> setting -> row
  for (const Row& r : rows)
    if (scale1(r)) bySet[str(r, "scene")][str(r, "setting")] = r;
  auto settingName = [](double mc, double fe) {
    char b[64];
    if (std::fabs(mc - 0.45) < 1e-9 && std::fabs(fe - 0.25) < 1e-9) return std::string("default");
    std::snprintf(b, sizeof b, "mc%.2f_fe%.*f", mc, std::fabs(fe - 0.25) < 1e-9 ? 2 : 1, fe);
    return std::string(b);
  };
  std::vector<Target> targets(4);
  targets[0].name = "Rebuild Detail (0 off / 1 on)";
  targets[0].values = {0, 1};
  targets[1].name = "Min Confidence";
  targets[1].values = {0.25, 0.35, 0.45, 0.55, 0.7};
  targets[2].name = "Region Feather %";
  targets[2].values = {0.0, 0.25, 0.5, 1.0, 2.0};
  targets[3].name = "Temporal Amount";
  targets[3].values = {0.0, 0.5, 1.0};
  std::map<std::string, double> regret;
  for (const std::string& s : scenes) {
    auto& m = bySet[s];
    auto sadOf = [&](const std::string& st) { return m.count(st) ? num(m[st], "sad") : NAN; };
    auto pick = [&](Target& T, const std::vector<double>& score, double tolAbs, double tolRel) {
      double b = 1e30;
      int bi = 0;
      for (size_t i = 0; i < score.size(); ++i)
        if (!std::isnan(score[i]) && score[i] < b) b = score[i], bi = int(i);
      const double tol = std::max(tolAbs, tolRel * b);
      for (size_t i = 0; i < score.size(); ++i)
        if (!std::isnan(score[i]) && score[i] <= b + tol) T.ok[s].insert(int(i));
      T.best[s] = bi;
    };
    double bestOn = 1e30;
    for (const auto& kv : m)
      if (num(kv.second, "rebuild") == 1 && (str(kv.second, "group") == "grid" || kv.first == "default"))
        bestOn = std::min(bestOn, num(kv.second, "sad"));
    pick(targets[0], {sadOf("rebuild_off"), bestOn}, 0.02, 0.05);
    std::vector<double> v;
    for (double mc : targets[1].values) v.push_back(sadOf(settingName(mc, 0.25)));
    pick(targets[1], v, 0.02, 0.05);
    v.clear();
    for (double fe : targets[2].values) v.push_back(sadOf(settingName(0.45, fe)));
    pick(targets[2], v, 0.02, 0.05);
    v = {num(m["tm0.0"], "edgeErr"), num(m["tm0.5"], "edgeErr"), num(m["default"], "edgeErr")};
    pick(targets[3], v, 0.0, 0.05);
    regret[s] = sadOf("default") - std::min(bestOn, sadOf("rebuild_off"));
  }
  const std::vector<std::string> measures = {"regionPct", "softness", "lumaP1", "lumaP99", "lumaRange", "baseCov",
                                             "pieces", "keptDefault", "anchored", "confMedian", "confP25",
                                             "confFracAbove45", "widthMedian", "jitterMeasured"};
  md << "| control | best value counts (over " << scenes.size() << " scenes) | constant rule: value, hit rate | stump rule "
     << "| stump hit rate | 2-fold hit rate |\n|---|---|---|---|---|---|\n";
  for (Target& T : targets) {
    std::map<int, int> cnt;
    for (const std::string& s : scenes) ++cnt[T.best[s]];
    std::string counts;
    for (const auto& kv : cnt) {
      char b[48];
      std::snprintf(b, sizeof b, "%s%g: %d", counts.empty() ? "" : ", ", T.values[size_t(kv.first)], kv.second);
      counts += b;
    }
    Stump cst;
    {  // constant only
      Stump all = fitStump(T, scenes, meas, {});
      cst = all;
    }
    const Stump st = fitStump(T, scenes, meas, measures);
    std::vector<std::string> even, odd;
    for (size_t i = 0; i < scenes.size(); ++i) (i % 2 ? odd : even).push_back(scenes[i]);
    const Stump se = fitStump(T, even, meas, measures), so = fitStump(T, odd, meas, measures);
    const int cv = evalStump(se, T, odd, meas) + evalStump(so, T, even, meas);
    char b[400];
    std::snprintf(b, sizeof b, "| %s | %s | %g, %.2f | %s < %.4g ? %g : %g | %.2f | %.2f |\n", T.name.c_str(),
                  counts.c_str(), T.values[size_t(cst.below)], double(cst.hits) / cst.n, st.measure.c_str(), st.thr,
                  T.values[size_t(st.below)], T.values[size_t(st.above)], double(st.hits) / st.n,
                  double(cv) / scenes.size());
    md << b;
  }
  {
    std::vector<double> rg;
    int big = 0;
    for (const auto& kv : regret) rg.push_back(kv.second), big += kv.second > 0.05;
    md << "\nThe default setting's SAD regret against the scene's best grid row: mean " << f3(mean(rg)) << "; scenes "
       << "with regret > 0.05: " << big << " of " << rg.size() << ".\n";
  }
  // The per-scene table (measurements + best values), for WP10.
  md << "\n### Per scene: measurements (centre frame) and the best values\n\n"
     << "| scene | region % | soft | luma p1 / p99 | base cov | pieces (kept) | conf median | width med | jitter | "
        "best rebuild | best mc | best feather | best tm | SAD default / best |\n"
     << "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
  for (const std::string& s : scenes) {
    const Row& m = meas[s];
    auto& r = bySet[s];
    char b[600];
    std::snprintf(b, sizeof b, "| %s | %.2f | %.2f | %.3f / %.3f | %.3f | %d (%d) | %.2f | %.1f | %.4f | %g | %g | %g | %g | %.3f / %.3f |\n",
                  s.c_str(), num(m, "regionPct"), num(m, "softness"), num(m, "lumaP1"), num(m, "lumaP99"),
                  num(m, "baseCov"), int(num(m, "pieces")), int(num(m, "keptDefault")), num(m, "confMedian"),
                  num(m, "widthMedian"), num(m, "jitterMeasured"), targets[0].values[size_t(targets[0].best[s])],
                  targets[1].values[size_t(targets[1].best[s])], targets[2].values[size_t(targets[2].best[s])],
                  targets[3].values[size_t(targets[3].best[s])], num(r["default"], "sad"),
                  num(r["default"], "sad") - regret[s]);
    md << b;
  }
  md << "\n## 8. Speed during the sweep (ms per frame through the host, 424x562, 2 cores)\n\n";
  for (const char* st : {"default", "vote_off", "tm_off", "vote_off_tm_off", "rebuild_off"}) {
    const Pred p = [&](const Row& r) { return ms(r) && str(r, "setting") == st; };
    md << "- " << st << ": " << f3(mean(col(rows, p, c("msPerFrame")))) << " ms\n";
  }
  std::printf("wrote %s/summary.md (%zu rows, %zu calibration scenes)\n", out.c_str(), rows.size(), scenes.size());
  return 0;
}
