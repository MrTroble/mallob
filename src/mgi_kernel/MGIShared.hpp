#ifndef _MGI_SHARED
#define _MGI_SHARED

typedef unsigned int m_uint;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __mgi_resolve_info {
    int literal;
    m_uint clauseOne;
    m_uint clauseTwo;
} MGIResolveInfo;

#ifdef __cplusplus
}
#endif

#endif
