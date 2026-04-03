#pragma once

#include <stdint.h>
#include <fstream>
#include <util/logger.hpp>
#include <string>
#include <vector>

namespace mgi
{
    struct TypeHandle
    {
        size_t internal = SIZE_MAX;

        inline operator bool()
        {
            return internal != SIZE_MAX;
        }
    };


    template<typename T>
    struct MGISpan
    {
        T* beginPtr = nullptr;
        T* endPtr = nullptr;

        MGISpan();

        template<typename G>
        MGISpan(const G& holder) : beginPtr(holder.data()), endPtr(holder.data() + holder.size()) {}

        size_t size() const {
            return size_t(endPtr - beginPtr);
        }

        T* begin() const {
            return beginPtr;
        }

        T* end() const {
            return endPtr;
        }

        T& operator[](size_t index){
            return beginPtr[index];
        }
    };
    
    template<typename T>
    using span = MGISpan<T>;

    struct Extension {
        void* extension = nullptr;
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

    template<typename T>
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
