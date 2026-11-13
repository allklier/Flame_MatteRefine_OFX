#include "recipe_core.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>

#include "merge_fusion.h"
#include "ops/edge_field.h"

namespace mr {
namespace recipes {

bool EvidenceKey::operator<(const EvidenceKey& o) const {
  return std::tie(kind, matteVersion, params) < std::tie(o.kind, o.matteVersion, o.params);
}

uint64_t paramHash(std::initializer_list<double> v) {
  uint64_t h = 1469598103934665603ull;
  for (double d : v) {
    uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    for (int i = 0; i < 8; ++i) {
      h ^= (b >> (8 * i)) & 0xffu;
      h *= 1099511628211ull;
    }
  }
  return h;
}

std::vector<float> Evidence::take() {
  if (pool_.empty()) return std::vector<float>();
  std::vector<float> v = std::move(pool_.back());
  pool_.pop_back();
  v.clear();  // keeps the capacity, drops the content
  return v;
}

void Evidence::giveBack(std::vector<float>&& v) {
  if (v.capacity()) pool_.push_back(std::move(v));
}

const std::vector<float>& Evidence::get(const EvidenceKey& k, const Compute& fn) {
  auto it = maps_.find(k);
  if (it != maps_.end()) {
    ++hits_;
    return it->second;
  }
  std::vector<float> v = take();
  fn(&v);  // a throw leaves no entry behind
  ++computed_;
  return maps_.emplace(k, std::move(v)).first->second;
}

void Evidence::dropMatteVersionsBelow(uint64_t v) {
  for (auto it = maps_.begin(); it != maps_.end();) {
    if (it->first.matteVersion != 0 && it->first.matteVersion < v) {
      giveBack(std::move(it->second));
      it = maps_.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<float>& Evidence::setPlane(const std::string& name, const float* data, size_t n) {
  auto it = planes_.find(name);
  if (it == planes_.end()) it = planes_.emplace(name, take()).first;
  it->second.assign(data, data + n);
  return it->second;
}

const std::vector<float>* Evidence::plane(const std::string& name) const {
  auto it = planes_.find(name);
  return it == planes_.end() ? nullptr : &it->second;
}

CandidatePlanes Evidence::newCandidate() {
  CandidatePlanes c;
  c.alpha = take();
  c.conf = take();
  return c;
}

void Evidence::addCandidate(CandidatePlanes&& c) { cand_.push_back(std::move(c)); }

void Evidence::beginRender() {
  for (auto& m : maps_) giveBack(std::move(m.second));
  for (auto& p : planes_) giveBack(std::move(p.second));
  for (auto& c : cand_) {
    giveBack(std::move(c.alpha));
    giveBack(std::move(c.conf));
  }
  maps_.clear();
  planes_.clear();
  cand_.clear();
  flows.reset();
  computed_ = hits_ = 0;
}

CandidatePlanes* Evidence::findCandidate(const std::string& id) {
  for (auto& c : cand_)
    if (c.id == id) return &c;
  return nullptr;
}

size_t Evidence::bytesHeld() const {
  size_t b = 0;
  for (const auto& m : maps_) b += m.second.size();
  for (const auto& p : planes_) b += p.second.size();
  for (const auto& c : cand_) b += c.alpha.size() + c.conf.size();
  return b * sizeof(float);
}

size_t Evidence::bytesPooled() const {
  size_t b = 0;
  for (const auto& v : pool_) b += v.capacity();
  return b * sizeof(float);
}

FrameStats computeStats(const FrameInputs& in) {
  FrameStats s;
  const size_t n = size_t(std::max(in.w, 0)) * std::max(in.h, 0);
  double sum = 0;
  long long grey = 0;
  for (size_t i = 0; i < n; ++i) {
    const float b = in.base[i];
    sum += b;
    grey += (b > 0.02f && b < 0.98f);
    if (in.region && in.region[i] > 0.0f) ++s.regionPixels;
  }
  s.baseMean = n ? sum / n : 0;
  s.greyFraction = n ? double(grey) / n : 0;
  return s;
}

const std::vector<float>& regionWeight(const StepContext& c, const RegionParams& p) {
  return c.ev.get({"regionWeight", 0, paramHash({p.growPx, p.featherPx})}, [&](std::vector<float>* v) {
    v->resize(size_t(c.in.w) * c.in.h);
    merge::prepareRegion(c.in.region, c.in.w, c.in.h, p.growPx, p.featherPx, v->data(), c.wk);
  });
}

Box regionBox(const StepContext& c, const RegionParams& p) {
  const std::vector<float>& b = c.ev.get({"regionBox", 0, paramHash({p.growPx, p.featherPx})}, [&](std::vector<float>* v) {
    const std::vector<float>& wgt = regionWeight(c, p);
    int x0 = c.in.w, y0 = c.in.h, x1 = 0, y1 = 0;
    for (int y = 0; y < c.in.h; ++y)
      for (int x = 0; x < c.in.w; ++x)
        if (wgt[size_t(y) * c.in.w + x] > 0.0f) {
          x0 = std::min(x0, x);
          x1 = std::max(x1, x + 1);
          y0 = std::min(y0, y);
          y1 = std::max(y1, y + 1);
        }
    *v = {float(x0), float(y0), float(x1), float(y1)};  // exact for any frame size below 2^24
  });
  Box r;
  r.x0 = int(b[0]);
  r.y0 = int(b[1]);
  r.x1 = int(b[2]);
  r.y1 = int(b[3]);
  return r;
}

const std::vector<float>& signedDistance(const StepContext& c, const float* matte, double band) {
  return c.ev.get({"sd0.5", c.matteVersion, paramHash({band})},
                  [&](std::vector<float>* v) { *v = signedEdgeDistance(matte, c.in.w, c.in.h, 0.5f, band); });
}

}  // namespace recipes
}  // namespace mr
