// macOS side of platform.h (decision 13), trimmed from LTX Alpha Gen's platform_macos.cpp, which
// was verified against the Command Line Tools MacOSX.sdk: host_statistics64(HOST_VM_INFO64) in
// mach/mach_host.h and mach/host_info.h, vm_statistics64 fields in mach/vm_statistics.h; sysctl
// names checked with `sysctl kern.osproductversion kern.osversion machdep.cpu.brand_string
// hw.ncpu hw.memsize`.
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>

#include <cstdint>
#include <cstdio>

#include "platform.h"

namespace mr {

namespace {
std::string gb(double bytes) {
  char b[32];
  std::snprintf(b, sizeof b, "%.1f GB", bytes / 1073741824.0);
  return b;
}

std::string sysctlString(const char* name) {
  char buf[256] = {0};
  size_t len = sizeof buf - 1;
  return sysctlbyname(name, buf, &len, nullptr, 0) == 0 ? buf : "unknown";
}

// free + inactive pages: what can be handed out without compressing or swapping.
double availableRamBytes() {
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm),
                        &count) != KERN_SUCCESS)
    return 0.0;
  return (double(vm.free_count) + double(vm.inactive_count)) * double(vm_kernel_page_size);
}
}  // namespace

std::vector<std::string> platformSnapshot() {
  std::vector<std::string> out;
  utsname u{};
  uname(&u);
  out.push_back("os: macOS " + sysctlString("kern.osproductversion") + " (" +
                sysctlString("kern.osversion") + "), Darwin " + u.release);
  int ncpu = 0;
  size_t len = sizeof ncpu;
  sysctlbyname("hw.ncpu", &ncpu, &len, nullptr, 0);
  out.push_back("cpu: " + sysctlString("machdep.cpu.brand_string") + ", " +
                std::to_string(ncpu) + " threads");
  uint64_t mem = 0;
  len = sizeof mem;
  sysctlbyname("hw.memsize", &mem, &len, nullptr, 0);
  out.push_back("ram: " + gb(double(mem)) + " total, " + gb(availableRamBytes()) + " available");
  return out;
}

}  // namespace mr
