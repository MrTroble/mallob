#pragma once

#include <stdint.h>
#include <fstream>
#include <util/logger.hpp>

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

    template <typename C>
    class OnExit
    {

        const C call;

    public:
        inline OnExit(const C cin) : call(cin) {}
        inline ~OnExit() { call(); }
    };

    inline std::string wholeFile(const std::string &path)
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
        std::string fileData;
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
