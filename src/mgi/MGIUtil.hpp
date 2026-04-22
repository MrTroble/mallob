#pragma once

#include <stdint.h>
#include <fstream>
#include <util/logger.hpp>
#include <string>
#include <vector>

// TODO: Use __VA_OPT__ here ... there is currently no statandart usage hear ... TO BAD!
// Change for C++20
#ifdef MGI_API_OCL
#define MGI_ERROR_CHECK(value, text, func, ...)                                                      \
    {                                                                                                \
        const auto err = (value);                                                                    \
        if (err != CL_SUCCESS)                                                                       \
        {                                                                                            \
            LOG(V0_CRIT, "[%s:%d] CL Error %d: " text "\n", __FILE__, __LINE__, err, ##__VA_ARGS__); \
            assert(false);                                                                           \
            func;                                                                                    \
        }                                                                                            \
    }
#endif
#ifdef DEBUG
#define MGI_DB_CHECK(value, text, ...) MGI_ERROR_CHECK(value, text, assert(false);, ##__VA_ARGS__)
#else
#define MGI_DB_CHECK(value, text, ...)
#endif

namespace mgi
{
    struct TypeHandle
    {
        size_t internal = SIZE_MAX;

        inline operator bool() const
        {
            return internal != SIZE_MAX;
        }

        template<typename T>
        inline bool operator==(const T& other)
        {
            return other.internal == internal;
        }
    };

    template <typename T>
    struct MGISpan
    {
        T *beginPtr = nullptr;
        T *endPtr = nullptr;

        MGISpan();

        template <typename G>
        MGISpan(const G &holder) : beginPtr(holder.data()), endPtr(holder.data() + holder.size()) {}

        size_t size() const
        {
            return size_t(endPtr - beginPtr);
        }

        T *begin() const
        {
            return beginPtr;
        }

        T *data() const
        {
            return beginPtr;
        }

        T *end() const
        {
            return endPtr;
        }

        T &operator[](size_t index)
        {
            return beginPtr[index];
        }

        bool empty()
        {
            return size() == 0;
        }

        T &back()
        {
            return *(end() - 1);
        }

        size_t size_bytes()
        {
            return size() * sizeof(T);
        }
    };

    template <typename T>
    using span = MGISpan<T>;

    struct Extension
    {
        void *extension = nullptr;
        size_t identiefier = 0;
    };

    template <typename C>
    class OnExit
    {

        const C call;

    public:
        inline OnExit(const C cin) : call(cin) {}
        inline ~OnExit() { call(); }
    };

    template <typename T>
    inline T wholeFile(const std::string &path)
    {
        std::ifstream inputstream(path,
                                  std::ios::ate | std::ios::in | std::ios::binary);
        if (!inputstream)
        {
            LOG(V1_WARN, "Error couldn't find file %s!\n", path.c_str());
            return {};
        }
        const size_t size = (size_t)inputstream.tellg();
        inputstream.seekg(0, std::ios_base::beg);
        T fileData;
        fileData.resize(size);
        inputstream.read((char *)fileData.data(), size);
        return fileData;
    }

    template <typename clType>
    struct Build
    {
        clType value[33];
        uint32_t current = 0;

        template <typename ValueType>
        inline Build<clType> &with(int pKey, const ValueType pValue)
        {
            value[current] = (clType)pKey;
            value[current + 1] = (clType)pValue;
            current += 2;
            value[current] = 0;
            return *this;
        }
    };

}

// TODO THIS IS STUPID REMOVE WITH C++20 Conecpts
#define MGI_DEFINE_TYPE_HASH(t1)                               \
    }                                                          \
    namespace std                                              \
    {                                                          \
        template <>                                            \
        struct hash<t1>                                        \
        {                                                      \
            std::size_t operator()(const t1 &s) const noexcept \
            {                                                  \
                return std::hash<size_t>{}(s.internal);        \
            }                                                  \
        };                                                     \
    }                                                          \
    namespace mgi                                              \
    {
