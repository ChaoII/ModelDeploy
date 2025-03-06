//
// Created by aichao on 2025/2/20.
//

#include "csrc/core/md_type.h"

namespace modeldeploy {
  int md_dtype_size(const MDDataType& data_type) {
    if (data_type == MDDataType::BOOL) {
      return sizeof(bool);
    } else if (data_type == MDDataType::INT16) {
      return sizeof(int16_t);
    } else if (data_type == MDDataType::INT32) {
      return sizeof(int32_t);
    } else if (data_type == MDDataType::INT64) {
      return sizeof(int64_t);
    } else if (data_type == MDDataType::FP32) {
      return sizeof(float);
    } else if (data_type == MDDataType::FP64) {
      return sizeof(double);
    } else if (data_type == MDDataType::UINT8) {
      return sizeof(uint8_t);
    } else if (data_type == MDDataType::INT8) {
      return sizeof(int8_t);
    } else {
      std::cerr<<"Unexpected data type: %s", str(data_type);
    }
    return -1;
  }

  std::string str(const MDDataType& fdt) {
    std::string out;
    switch (fdt) {
    case MDDataType::BOOL:
      out = "MDDataType::BOOL";
      break;
    case MDDataType::INT16:
      out = "MDDataType::INT16";
      break;
    case MDDataType::INT32:
      out = "MDDataType::INT32";
      break;
    case MDDataType::INT64:
      out = "MDDataType::INT64";
      break;
    case MDDataType::FP32:
      out = "MDDataType::FP32";
      break;
    case MDDataType::FP64:
      out = "MDDataType::FP64";
      break;
    case MDDataType::UINT8:
      out = "MDDataType::UINT8";
      break;
    case MDDataType::INT8:
      out = "MDDataType::INT8";
      break;
    default:
      out = "MDDataType::UNKNOWN";
    }
    return out;
  }

  std::ostream& operator<<(std::ostream& out, const MDDataType& md_dtype) {
    switch (md_dtype) {
    case MDDataType::BOOL:
      out << "MDDataType::BOOL";
      break;
    case MDDataType::INT16:
      out << "MDDataType::INT16";
      break;
    case MDDataType::INT32:
      out << "MDDataType::INT32";
      break;
    case MDDataType::INT64:
      out << "MDDataType::INT64";
      break;
    case MDDataType::FP32:
      out << "MDDataType::FP32";
      break;
    case MDDataType::FP64:
      out << "MDDataType::FP64";
      break;
    case MDDataType::UINT8:
      out << "MDDataType::UINT8";
      break;
    case MDDataType::INT8:
      out << "MDDataType::INT8";
      break;
    default:
      out << "MDDataType::UNKNOWN";
    }
    return out;
  }

  template <typename PlainType>
  const MDDataType TypeToDataType<PlainType>::dtype = UNKNOWN1;

  template <>
  const MDDataType TypeToDataType<bool>::dtype = BOOL;

  template <>
  const MDDataType TypeToDataType<int16_t>::dtype = INT16;

  template <>
  const MDDataType TypeToDataType<int32_t>::dtype = INT32;

  template <>
  const MDDataType TypeToDataType<int64_t>::dtype = INT64;

  template <>
  const MDDataType TypeToDataType<float>::dtype = FP32;

  template <>
  const MDDataType TypeToDataType<double>::dtype = FP64;

  template <>
  const MDDataType TypeToDataType<uint8_t>::dtype = UINT8;

  template <>
  const MDDataType TypeToDataType<int8_t>::dtype = INT8;
}