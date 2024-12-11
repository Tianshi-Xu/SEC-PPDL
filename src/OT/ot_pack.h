#define OT_TYPES 8 

class OTPack {
public:
    int ot_tag;
    void *ot_extension_n[OT_TYPES]; // 可以指向 SilentOT 或 SplitIKNP
    void *ot_extension_1;

    OTPack(int tag) : ot_tag(tag), ot_extension(nullptr) {
        // 将 ot_extension_n 初始化为 nullptr
        for (int i = 0; i < OT_TYPES; i++) {
            ot_extension_n[i] = nullptr;
        }

        if (ot_tag == 1) {
            // 当 ot_tag == 1，ot_extension 是 cheetah::SilentOT<NetIO>*
            ot_extension_1 = new cheetah::SilentOT<NetIO>();
            for (int i = 0; i < OT_TYPES; i++) {
                ot_extension_n[i] = new cheetah::SilentOTN<NetIO>(silent_ot, 1 << (i + 1));
            }
        } else {
            // 当 ot_tag != 1，ot_extension 是 SplitIKNP<NetIO>*
            ot_extension_1 = new SplitIKNP<NetIO>();
            for (int i = 0; i < OT_TYPES; i++) {
                ot_extension_n[i] = new SplitKKOT<NetIO>(party, iopack->io, 1 << (i + 1));
            }
        }
    }

};