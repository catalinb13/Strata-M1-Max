// metal/tests/window_cells_test.cpp - the park's K/V repack shuffle, model-free: the image holds exactly the
// bytes of the window (round-trip identity), nothing is written past it (the [cell]->[page] index mix-up that
// wrote 4x past the scratch is the bug this catches), and every unit lands at its page/head/in-page address.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// The real shuffle from src/core/conversation_snapshot.cpp (built into strata_metal_host).
namespace strata::core {
void window_cells(void* dst, const void* src, int64_t cells, int64_t page, int64_t heads, size_t unit,
                  bool to_image);
}

namespace {

int fails = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    ++fails;
}
void one(int64_t cells, int64_t page, int64_t heads, size_t unit, std::mt19937& rng) {
    const int64_t pages = (cells + page - 1) / page;
    std::vector<uint8_t> win((size_t) cells * heads * unit);
    for (auto& b : win) b = static_cast<uint8_t>(rng());
    const size_t img_size = (size_t) pages * heads * page * unit;
    const size_t pad = 64;
    std::vector<uint8_t> buf(pad + img_size + pad, 0xA5);
    uint8_t* img = buf.data() + pad;

    strata::core::window_cells(img, win.data(), cells, page, heads, unit, true);
    check(std::memcmp(buf.data(), std::vector<uint8_t>(pad, 0xA5).data(), pad) == 0 &&
          std::memcmp(buf.data() + pad + img_size, std::vector<uint8_t>(pad, 0xA5).data(), pad) == 0,
          "park wrote outside the image");
    for (int64_t pg = 0, p = 0; p < cells; ++pg, p += page)
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t r = 0; r < page && p + r < cells; ++r) {
                const size_t wi = (size_t) (((p + r) * heads + h) * unit);
                const size_t ii = (size_t) (((pg * heads + h) * page + r) * unit);
                check(std::memcmp(img + ii, win.data() + wi, unit) == 0, "unit at wrong image address");
                if (fails) return;
            }
    std::vector<uint8_t> back(win.size(), 0);
    strata::core::window_cells(back.data(), img, cells, page, heads, unit, false);
    check(back == win, "restore is not the inverse of the park");
}

// The incremental park conversation_kv_save's Metal branch does: a whole number of pages of the prefix is kept
// from the previous park and only cells [keep, cells) are repacked into a tail-sized scratch, written at the
// prefix offset.  It must land on the same bytes a full pack writes - which needs the image's page order to
// group by page (the prefix is then contiguous and equal to `img_size / cells * keep` bytes, the same per-cell
// stride the window order has) - and must not touch one byte before that offset.
void tail_pack(int64_t cells, int64_t page, int64_t keep, int64_t heads, size_t unit) {
    check(keep % page == 0 && cells % page == 0, "tail_pack needs whole pages");
    if (keep % page != 0 || cells % page != 0) return;
    const size_t img_size = (size_t) cells * heads * unit;
    std::vector<uint8_t> win((size_t) cells * heads * unit);
    for (size_t b = 0; b < win.size(); ++b)
        win[b] = static_cast<uint8_t>(b * 31u + (b >> 8) * 7u + 1u);   // deterministic, changes every byte
    std::vector<uint8_t> ref(img_size, 0);
    strata::core::window_cells(ref.data(), win.data(), cells, page, heads, unit, true);

    const size_t prefix = img_size / (size_t) cells * (size_t) keep;   // == heads * unit * keep, bytes per cell * keep
    std::vector<uint8_t> stale(img_size, 0);                            // what the previous park's tail held
    for (size_t b = 0; b < img_size; ++b) stale[b] = static_cast<uint8_t>(b * 13u + 0x5Cu);
    std::memcpy(stale.data(), ref.data(), prefix);                      // the retained prefix is the last park's bytes

    std::vector<uint8_t> img = stale;                                   // the image the incremental pack starts from
    std::vector<uint8_t> scr(img_size - prefix);
    strata::core::window_cells(scr.data(), win.data() + prefix, cells - keep, page, heads, unit, true);
    std::memcpy(img.data() + prefix, scr.data(), scr.size());
    check(img == ref, "tail-only pack differs from the full pack");
    if (fails) return;
    // the tail scratch must be exactly the reference's bytes from the prefix on (the scratch itself, not just the
    // merged image: catches a scratch sized or offset wrong in a way the merge hides)
    check(std::memcmp(scr.data(), ref.data() + prefix, scr.size()) == 0, "tail scratch != reference tail");

    // the same scratch written at `prefix` into an image whose every byte is 0xAA (a byte the pattern cannot
    // produce) leaves the 0xAA below `prefix` in place: the tail-only pack touches nothing before it
    std::vector<uint8_t> poison(img_size, 0xAA);
    std::memcpy(poison.data() + prefix, scr.data(), scr.size());
    bool prefix_untouched = true;
    for (size_t b = 0; b < prefix; ++b) prefix_untouched = prefix_untouched && poison[b] == 0xAA;
    check(prefix_untouched, "tail-only pack wrote before the prefix");
    check(std::memcmp(poison.data() + prefix, ref.data() + prefix, img_size - prefix) == 0,
          "tail-only pack's tail differs from the reference tail");
}

}  // namespace

int main() {
    std::mt19937 rng(20261019);
    one(5, 4, 2, 3, rng);      // trailing partial page, tiny unit
    one(0, 4, 2, 3, rng);      // nothing to park
    one(64, 16, 8, 512, rng);  // int8 cells: 512-byte unit, whole pages
    one(100, 16, 2, 512, rng); // partial last page at the real cell size
    one(33, 1, 3, 17, rng);    // page = 1: the shuffle is the identity
    // the incremental park (keep a whole number of pages, as conversation_kv_save floors it to)
    tail_pack(1024, 16, 0, 8, 512);      // nothing reused: the cold copy, byte-identical to the full pack
    tail_pack(1024, 16, 512, 8, 512);    // half the pages reused, at the real int8 cell size
    tail_pack(1024, 16, 1008, 8, 512);   // one page of tail - what one new turn of a parked conversation packs
    tail_pack(1024, 16, 1024, 8, 512);   // empty tail: a park that reuses everything writes nothing
    tail_pack(64, 1, 33, 3, 17);         // page = 1, tiny unit
    tail_pack(4096, 16, 1024, 2, 128);   // fp16-sized unit over a long parked conversation
    if (fails) { std::printf("window_cells_test: %d FAILURES\n", fails); return 1; }
    std::printf("window_cells_test: PASS\n");
    return 0;
}
