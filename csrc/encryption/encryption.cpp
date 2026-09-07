#include <fstream>
#include <vector>
#include <cstring>
#include "utils/utils.h"
#include "encryption/encryption.h"

#ifdef ENABLE_ENCRYPTION
#include <mbedtls/gcm.h>
#include <mbedtls/pkcs5.h>
#include <mbedtls/md.h>
#include <mbedtls/entropy.h>
#include <mbedtls/ctr_drbg.h>

namespace modeldeploy {

    // ==================== 常量（文件格式 V3） ====================
    // [4]   魔数 "MDEN"
    // [4]   版本号 (3)
    // [4]   格式字符串长度
    // [N]   格式字符串
    // [16]  Salt（PBKDF2 key 派生用）
    // [12]  GCM nonce
    // [4]   密文长度
    // [N]   密文（AES-256-GCM）
    // [16]  GCM 认证标签（128-bit，替代旧 CRC，防篡改）
    constexpr static uint32_t SALT_LEN = 16;
    constexpr static uint32_t NONCE_LEN = 12;
    constexpr static uint32_t TAG_LEN = 16;
    constexpr static uint32_t AES_KEY_LEN = 32;
    constexpr static uint32_t PBKDF2_ITERATIONS = 100000;
    constexpr static uint32_t VERSION = 3;
    const static std::string MAGIC = "MDEN";

    static bool is_password_invalid(const std::string& pwd) { return pwd.empty(); }

    // ==================== PBKDF2-HMAC-SHA256 密钥派生 ====================
    static bool derive_key(const std::string& password,
                           const uint8_t* salt, uint32_t salt_len,
                           uint8_t* out_key) {
        return mbedtls_pkcs5_pbkdf2_hmac_ext(MBEDTLS_MD_SHA256,
                    reinterpret_cast<const unsigned char*>(password.data()), password.size(),
                    salt, salt_len, PBKDF2_ITERATIONS, AES_KEY_LEN, out_key) == 0;
    }

    // ==================== AES-256-GCM 加密 ====================
    static bool gcm_encrypt(const uint8_t* key, const uint8_t* nonce,
                            const uint8_t* plain, uint32_t plain_len,
                            std::vector<uint8_t>* cipher, uint8_t* tag) {
        mbedtls_gcm_context ctx;
        mbedtls_gcm_init(&ctx);
        bool ok = false;
        if (mbedtls_gcm_setkey(&ctx, MBEDTLS_CIPHER_ID_AES, key, 256) == 0) {
            cipher->resize(plain_len);
            ok = mbedtls_gcm_crypt_and_tag(&ctx, MBEDTLS_GCM_ENCRYPT, plain_len,
                    nonce, NONCE_LEN, nullptr, 0,
                    plain, cipher->data(), TAG_LEN, tag) == 0;
        }
        mbedtls_gcm_free(&ctx);
        return ok;
    }

    // ==================== AES-256-GCM 解密（认证） ====================
    static bool gcm_decrypt(const uint8_t* key, const uint8_t* nonce,
                            const uint8_t* cipher_data, uint32_t cipher_len,
                            const uint8_t* tag, std::vector<uint8_t>* plain) {
        mbedtls_gcm_context ctx;
        mbedtls_gcm_init(&ctx);
        bool ok = false;
        if (mbedtls_gcm_setkey(&ctx, MBEDTLS_CIPHER_ID_AES, key, 256) == 0) {
            plain->resize(cipher_len);
            ok = mbedtls_gcm_auth_decrypt(&ctx, cipher_len,
                    nonce, NONCE_LEN, nullptr, 0,
                    tag, TAG_LEN, cipher_data, plain->data()) == 0;
        }
        mbedtls_gcm_free(&ctx);
        return ok;
    }

    // ==================== 随机数（熵源 + CTR-DRBG） ====================
    static bool fill_random(uint8_t* out, uint32_t len) {
        mbedtls_entropy_context entropy;
        mbedtls_ctr_drbg_context drbg;
        mbedtls_entropy_init(&entropy);
        mbedtls_ctr_drbg_init(&drbg);
        bool ok = (mbedtls_ctr_drbg_seed(&drbg, mbedtls_entropy_func, &entropy, nullptr, 0) == 0) &&
                  (mbedtls_ctr_drbg_random(&drbg, out, len) == 0);
        mbedtls_ctr_drbg_free(&drbg);
        mbedtls_entropy_free(&entropy);
        return ok;
    }

    // ==================== CRC32（保留兼容既有外部调用） ====================
    uint32_t calculate_crc32(const std::string& data) {
        static const uint32_t table[256] = {
            0x00000000L, 0x77073096L, 0xee0e612cL, 0x990951baL, 0x076dc419L, 0x706af48fL,
            0xe963a535L, 0x9e6495a3L, 0x0edb8832L, 0x79dcb8a4L, 0xe0d5e91eL, 0x97d2d988L,
            0x09b64c2bL, 0x7eb17cbdL, 0xe7b82d07L, 0x90bf1d91L, 0x1db71064L, 0x6ab020f2L,
            0xf3b97148L, 0x84be41deL, 0x1adad47dL, 0x6ddde4ebL, 0xf4d4b551L, 0x83d385c7L,
        };
        uint32_t crc = 0xFFFFFFFF;
        for (const char c : data)
            crc = table[(crc ^ (uint8_t)c) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFF;
    }

    // ==================== 文件格式（V3） ====================
    bool encrypt_model_file(const std::string& input_path, const std::string& output_path,
                            const std::string& password, const std::string& model_format) {
        if (is_password_invalid(password)) { MD_LOG_ERROR << "Password cannot be empty." << std::endl; return false; }

        std::string model_data;
        if (!read_binary_from_file(input_path, &model_data))
        { MD_LOG_ERROR << "Failed to read model file: " << input_path << std::endl; return false; }

        uint8_t salt[SALT_LEN], nonce[NONCE_LEN], tag[TAG_LEN];
        if (!fill_random(salt, SALT_LEN) || !fill_random(nonce, NONCE_LEN))
        { MD_LOG_ERROR << "Failed to generate random bytes." << std::endl; return false; }

        uint8_t aes_key[AES_KEY_LEN];
        if (!derive_key(password, salt, SALT_LEN, aes_key))
        { MD_LOG_ERROR << "Key derivation failed." << std::endl; return false; }

        std::vector<uint8_t> cipher;
        if (!gcm_encrypt(aes_key, nonce, (const uint8_t*)model_data.data(),
                         (uint32_t)model_data.size(), &cipher, tag))
        { MD_LOG_ERROR << "AES-GCM encryption failed." << std::endl; return false; }

        std::ofstream out(output_path, std::ios::binary);
        if (!out.is_open())
        { MD_LOG_ERROR << "Failed to create encrypted file: " << output_path << std::endl; return false; }

        auto w32 = [&](uint32_t v) { out.write((const char*)&v, 4); };
        out.write(MAGIC.data(), 4); w32(VERSION);
        w32((uint32_t)model_format.size()); out.write(model_format.data(), (std::streamsize)model_format.size());
        out.write((const char*)salt, SALT_LEN); out.write((const char*)nonce, NONCE_LEN);
        w32((uint32_t)cipher.size()); out.write((const char*)cipher.data(), (std::streamsize)cipher.size());
        out.write((const char*)tag, TAG_LEN);
        out.close();
        MD_LOG_INFO << "Model encrypted: " << output_path << std::endl;
        return true;
    }

    static bool read_header(const std::string& path, std::string* fmt,
                            std::vector<uint8_t>* salt, std::vector<uint8_t>* nonce,
                            std::vector<uint8_t>* cipher, std::vector<uint8_t>* tag) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) { MD_LOG_ERROR << "Cannot open: " << path << std::endl; return false; }

        auto r32 = [&](uint32_t* v) { return (bool)in.read((char*)v, 4); };
        char magic[4]; uint32_t ver, fmt_len, data_len;
        if (!in.read(magic, 4) || std::string(magic, 4) != MAGIC)
        { MD_LOG_ERROR << "Bad magic." << std::endl; return false; }
        if (!r32(&ver)) return false;
        if (ver != VERSION) {
            MD_LOG_ERROR << "Unsupported version: " << ver
                         << " (only V" << VERSION << " supported; older files are deprecated)." << std::endl;
            return false;
        }
        if (!r32(&fmt_len)) return false;
        fmt->resize(fmt_len); if (!in.read(&(*fmt)[0], fmt_len)) return false;

        salt->resize(SALT_LEN); nonce->resize(NONCE_LEN); tag->resize(TAG_LEN);
        if (!in.read((char*)salt->data(), SALT_LEN) || !in.read((char*)nonce->data(), NONCE_LEN)) return false;

        if (!r32(&data_len)) return false;
        cipher->resize(data_len); if (!in.read((char*)cipher->data(), data_len)) return false;
        if (!in.read((char*)tag->data(), TAG_LEN)) return false;
        return true;
    }

    bool decrypt_model_file(const std::string& in_path, const std::string& out_path,
                            const std::string& password) {
        if (is_password_invalid(password)) { MD_LOG_ERROR << "Password cannot be empty." << std::endl; return false; }
        std::string fmt; std::vector<uint8_t> salt, nonce, cipher, tag;
        if (!read_header(in_path, &fmt, &salt, &nonce, &cipher, &tag)) return false;

        uint8_t key[AES_KEY_LEN];
        if (!derive_key(password, salt.data(), (uint32_t)salt.size(), key))
        { MD_LOG_ERROR << "Key derivation failed." << std::endl; return false; }

        std::vector<uint8_t> plain;
        if (!gcm_decrypt(key, nonce.data(), cipher.data(), (uint32_t)cipher.size(), tag.data(), &plain))
        { MD_LOG_ERROR << "Decryption failed: wrong password or corrupted file." << std::endl; return false; }

        std::ofstream out(out_path, std::ios::binary);
        if (!out.is_open()) { MD_LOG_ERROR << "Cannot create: " << out_path << std::endl; return false; }
        out.write((const char*)plain.data(), (std::streamsize)plain.size());
        MD_LOG_INFO << "Model decrypted: " << out_path << std::endl;
        return true;
    }

    bool is_encrypted_model_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        char m[4]; f.read(m, 4);
        return std::string(m, 4) == MAGIC;
    }

    std::string get_model_format_from_encrypted_file(const std::string& path) {
        std::string fmt; std::vector<uint8_t> s, n, c, t;
        read_header(path, &fmt, &s, &n, &c, &t); return fmt;
    }

    bool read_encrypted_model_to_buffer(const std::string& path, const std::string& password,
                                        std::string* buf, std::string* fmt) {
        if (!buf || !fmt) return false;
        if (is_password_invalid(password)) return false;
        std::vector<uint8_t> salt, nonce, cipher, tag;
        if (!read_header(path, fmt, &salt, &nonce, &cipher, &tag)) return false;
        uint8_t key[AES_KEY_LEN];
        if (!derive_key(password, salt.data(), (uint32_t)salt.size(), key)) { buf->clear(); fmt->clear(); return false; }
        std::vector<uint8_t> plain;
        if (!gcm_decrypt(key, nonce.data(), cipher.data(), (uint32_t)cipher.size(), tag.data(), &plain)) { buf->clear(); fmt->clear(); return false; }
        buf->assign((const char*)plain.data(), plain.size());
        return true;
    }
} // namespace modeldeploy
#endif // ENABLE_ENCRYPTION
