#ifndef _MGI_SHARED
#define _MGI_SHARED

typedef unsigned int m_uint;

#ifdef __cplusplus
#include <atomic>
#include <cmath>
    using namespace std;

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

    typedef std::atomic<float> atomic_float;

    #define MGI_K_INLINE inline
#else
    #define MGI_K_INLINE
#endif

    #ifdef MGI_API_OCL
    #define MGI_GLOBAL global 
    #define MGI_SHARED local 
    #define MGI_CONST const constant
    #define MGI_LOCAL private 

    #define MGI_BARRIER(flags) barrier(flags)
    #define MGI_MEM_GLOBAL CLK_GLOBAL_MEM_FENCE
    #define MGI_MEM_LOCAL CLK_LOCAL_MEM_FENCE
    #define MGI_MEM_IMAGE CLK_IMAGE_MEM_FENCE

    #else
    #define MGI_GLOBAL  
    #define MGI_SHARED  
    #define MGI_CONST 
    #define MGI_LOCAL  
        
    #define MGI_BARRIER(flags)
    #define MGI_MEM_GLOBAL 1
    #define MGI_MEM_LOCAL 2
    #define MGI_MEM_IMAGE 4
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

MGI_K_INLINE void mgiRNGInit(MGIRng* rng, m_uint x, m_uint y, m_uint id, m_uint seedVal) {
    rng->seed.x = x ^ (id << 16);
    rng->seed.y = y ^ ((id + seedVal) << 16);
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

typedef struct __mgi_atomic_reservoir {
    MGIResolveInfo resolve;
    atomic_float weight;
} MGIAtomicReservoir;

typedef struct __mgi_reservoir {
    MGIResolveInfo resolve;
    float weight;
} MGIReservoir;

#define MGI_MAX_DEBUG_MESSAGE_SPACE 1024

typedef struct __mgi_debug_helper {
    char messageBuffer[MGI_MAX_DEBUG_MESSAGE_SPACE];
    m_uint lastIndex;
    m_uint overflowMessageCount;
    atomic_flag flag;
} MGIDebugHelper;

typedef struct __mgi_info {
    m_uint maxClauseSize;
    MGIDebugHelper __pDebugHelper;
} MGIInfo;

MGI_K_INLINE void __internal_print(MGI_GLOBAL MGIDebugHelper* helper, MGI_CONST char* message) {
    while(!atomic_flag_test_and_set(&helper->flag));
    while (*message != 0 && helper->lastIndex < MGI_MAX_DEBUG_MESSAGE_SPACE)
    {
        if(helper->lastIndex >= MGI_MAX_DEBUG_MESSAGE_SPACE) {
            helper->overflowMessageCount++;
        } else {
            helper->messageBuffer[helper->lastIndex++] = *message;
            message++;
        }
    }
    if(helper->lastIndex <= MGI_MAX_DEBUG_MESSAGE_SPACE) {
        helper->messageBuffer[helper->lastIndex++] = '\n';
    }
    if(helper->lastIndex <= MGI_MAX_DEBUG_MESSAGE_SPACE) {
        helper->messageBuffer[helper->lastIndex] = 0;
    }
    atomic_flag_clear(&helper->flag);
}

#define MGI_DEBUG_LOG(message) __internal_print(&info->__pDebugHelper, (MGI_CONST char*)(message))

MGI_K_INLINE void mgiAtomicReserviorAddSample(MGI_GLOBAL MGIAtomicReservoir* reservior, MGIRng* rng, const MGIResolveInfo* resolve, float weight) {
    float value = atomic_load(&reservior->weight);
    while(atomic_compare_exchange_strong(&reservior->weight, &value, value + weight) != value)
        value = atomic_load(&reservior->weight);
    // Can we get rid of those compare exchanges
    // Does this kill the probabilty? Look at the Markov Chain: Ergodisity even needed?
    if((weight / (value + weight)) >= mgiRNGRndFloat(rng)) {
        reservior->resolve = *resolve;
    }
}

MGI_K_INLINE void mgiReserviorAddSample(MGIReservoir* reservior, MGIRng* rng, const MGIResolveInfo* resolve, float weight) {
    reservior->weight += weight;
    if((weight / reservior->weight) >= mgiRNGRndFloat(rng)) {
        reservior->resolve = *resolve;
    }
}


#ifdef __cplusplus
}
#endif

#endif
