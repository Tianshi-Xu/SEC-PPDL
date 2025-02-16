#pragma once

namespace Datatype {

enum LOCATION { HOST = 0, DEVICE, HOST_AND_DEVICE, UNDEF };

class UnifiedBase {
public:
  virtual bool on_host() const { return loc_ == HOST; }

  virtual bool on_device() const { return loc_ == DEVICE; }

private:
  LOCATION loc_;
};

} // namespace Datatype