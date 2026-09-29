#pragma once

#include <rapidjson/document.h>

#include <string>
#include <type_traits>

namespace opencalibration
{

template <typename T> bool readJsonValue(const rapidjson::Value &v, T &out)
{
    if constexpr (std::is_same_v<T, std::string>)
    {
        if (!v.IsString())
            return false;
        out = v.GetString();
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        if (!v.IsNumber())
            return false;
        out = v.GetDouble();
    }
    else
    {
        if (!v.template Is<T>())
            return false;
        out = v.template Get<T>();
    }
    return true;
}

template <typename T> bool readJsonField(const rapidjson::Value &obj, const char *name, T &out)
{
    if (!obj.IsObject())
        return false;
    const auto it = obj.FindMember(name);
    return it != obj.MemberEnd() && readJsonValue(it->value, out);
}

template <typename Vector> bool readJsonArrayField(const rapidjson::Value &obj, const char *name, Vector &out)
{
    if (!obj.IsObject())
        return false;
    const auto it = obj.FindMember(name);
    if (it == obj.MemberEnd() || !it->value.IsArray())
        return false;
    const auto &arr = it->value;
    for (rapidjson::SizeType i = 0; i < arr.Size() && i < static_cast<rapidjson::SizeType>(out.size()); i++)
        readJsonValue(arr[i], out[i]);
    return true;
}

inline const rapidjson::Value *findJsonMember(const rapidjson::Value &obj, const char *name)
{
    if (!obj.IsObject())
        return nullptr;
    const auto it = obj.FindMember(name);
    return it == obj.MemberEnd() ? nullptr : &it->value;
}

} // namespace opencalibration
