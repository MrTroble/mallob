#ifndef _MGI_SHARED
#define _MGI_SHARED

typedef unsigned int m_uint;

#ifdef __cplusplus
#include <atomic>
extern "C" {
    struct uint2 {
        m_uint x;
        m_uint y;

        inline uint2 operator+(m_uint l) { return { x + l, y + l}; }
        inline uint2 operator>>(m_uint l) { return { x >> l, y >> l}; }
        inline uint2 operator^(uint2 l) { return { x ^ l.x, y ^ l.y}; }
        inline uint2& operator^=(uint2 l) { x ^= l.x; y ^= l.y; return *this; }
    };
    inline uint2 operator*(m_uint l, uint2 o) { return { o.x*l, o.y*l}; }
    using namespace std;

    typedef std::atomic<float> atomic_float;

    #define MGI_K_INLINE inline
#else
    #define MGI_K_INLINE
#endif
typedef uint2 m_uint2;

typedef struct __mgi_rng {
    m_uint2 seed;
} MGIRng;

MGI_K_INLINE float mgiUintToFloat(m_uint x) {
    union {
        unsigned int a;
        float b;
    } u;
    u.a = 0x3f800000 | (x >> 9);
    return u.b - 1.f;
}

// returns random float between (0,1]
MGI_K_INLINE float mgiRNGRndFloat(MGIRng* rng) {
    // PCG2D, as described here: https://jcgt.org/published/0009/03/02/
    rng->seed = 1664525u * rng->seed + 1013904223u;
    rng->seed.x += 1664525u * rng->seed.y;
    rng->seed.y += 1664525u * rng->seed.x;
    rng->seed ^= (rng->seed >> 16u);
    rng->seed.x += 1664525u * rng->seed.y;
    rng->seed.y += 1664525u * rng->seed.x;
    rng->seed ^= (rng->seed >> 16u);
    // return float from seed
    return 1.f - mgiUintToFloat(rng->seed.x);
}

typedef struct __mgi_resolve_info {
    int literal;
    m_uint clauseOne;
    m_uint clauseTwo;
    m_uint resolvedSize;
} MGIResolveInfo;

typedef struct __mgi_reservoir {
    MGIResolveInfo resolve;
    atomic_float weight;
} MGIReservoir;
#ifdef __cplusplus
}
#endif

#endif
