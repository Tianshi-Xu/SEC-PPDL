#pragma once

namespace Datatype {

enum LOCATION { HOST = 0, DEVICE, HOST_AND_DEVICE, UNDEF };

class UnifiedBase {
public:
  UnifiedBase(LOCATION loc = UNDEF) : loc_(loc){};

  UnifiedBase(const UnifiedBase &) = default;

  UnifiedBase &operator=(const UnifiedBase &) = default;

  UnifiedBase(UnifiedBase &&) = default;

  UnifiedBase &operator=(UnifiedBase &&) = default;

  virtual bool on_host() const { return loc_ == HOST; }

  virtual bool on_device() const { return loc_ == DEVICE; }

  LOCATION location() const { return loc_; }

protected:
  LOCATION loc_;
};

} // namespace Datatype