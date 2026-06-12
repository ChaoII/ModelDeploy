#include <fstream>
#include <vector>
#include <cstring>
#include "utils/utils.h"
#include "encryption/encryption.h"

#ifdef ENABLE_ENCRYPTION
#include <openssl/evp.h>
#include <openssl/rand.h>

namespace modeldeploy {

    // ==================== SHA-256 密钥派生 ====================
    static void derive_key(const std::string& password,
                           const uint8_t* salt, uint32_t salt_len,
                           uint8_t* out_key) {
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
        EVP_DigestUpdate(ctx, password.data(), password.size());
        EVP_DigestUpdate(ctx, salt, salt_len);
        EVP_DigestFinal_ex(ctx, out_key, nullptr);
        EVP_MD_CTX_free(ctx);
    }

    // ==================== AES-256-CBC 加密 ====================
    static bool aes_encrypt(const uint8_t* key, const uint8_t* iv,
                            const uint8_t* plain, uint32_t plain_len,
                            std::vector<uint8_t>* cipher) {
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx) return false;
        bool ok = false;
        if (EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, key, iv) == 1) {
            cipher->resize(plain_len + 32);
            int out_len = 0;
            if (EVP_EncryptUpdate(ctx, cipher->data(), &out_len, plain, (int)plain_len) == 1) {
                int tail = 0;
                ok = (EVP_EncryptFinal_ex(ctx, cipher->data() + out_len, &tail) == 1);
                cipher->resize((size_t)out_len + tail);
            }
        }
        EVP_CIPHER_CTX_free(ctx);
        return ok;
    }

    static bool aes_decrypt(const uint8_t* key, const uint8_t* iv,
                            const uint8_t* cipher_data, uint32_t cipher_len,
                            std::vector<uint8_t>* plain) {
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx) return false;
        bool ok = false;
        if (EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, key, iv) == 1) {
            plain->resize(cipher_len + 32);
            int out_len = 0;
            if (EVP_DecryptUpdate(ctx, plain->data(), &out_len, cipher_data, (int)cipher_len) == 1) {
                int tail = 0;
                ok = (EVP_DecryptFinal_ex(ctx, plain->data() + out_len, &tail) == 1);
                plain->resize((size_t)out_len + tail);
            }
        }
        EVP_CIPHER_CTX_free(ctx);
        return ok;
    }

    // ==================== CRC32 ====================
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

    // ==================== 文件格式（V2） ====================
    // [4] 魔数 "MDEN"
    // [4] 版本号 (2)
    // [4] 格式字符串长度
    // [N] 格式字符串
    // [16] Salt（密钥派生用）
    // [16] IV（AES-CBC）
    // [4] 密文长度
    // [N] 密文
    // [4] CRC32（对密文校验）

    constexpr static uint32_t SALT_LEN = 16;
    constexpr static uint32_t IV_LEN = 16;
    constexpr static uint32_t AES_KEY_LEN = 32;
    constexpr static uint32_t VERSION = 2;
    const static std::string MAGIC = "MDEN";

    static bool is_password_invalid(const std::string& pwd) { return pwd.empty(); }

    bool encrypt_model_file(const std::string& input_path, const std::string& output_path,
                            const std::string& password, const std::string& model_format) {
        if (is_password_invalid(password)) { MD_LOG_ERROR << "Password cannot be empty." << std::endl; return false; }

        std::string model_data;
        if (!read_binary_from_file(input_path, &model_data))
        { MD_LOG_ERROR << "Failed to read model file: " << input_path << std::endl; return false; }

        uint8_t salt[SALT_LEN], iv[IV_LEN];
        if (RAND_bytes(salt, SALT_LEN) != 1 || RAND_bytes(iv, IV_LEN) != 1)
        { MD_LOG_ERROR << "Failed to generate random bytes." << std::endl; return false; }

        uint8_t aes_key[AES_KEY_LEN];
        derive_key(password, salt, SALT_LEN, aes_key);

        std::vector<uint8_t> cipher;
        if (!aes_encrypt(aes_key, iv, (const uint8_t*)model_data.data(), (uint32_t)model_data.size(), &cipher))
        { MD_LOG_ERROR << "AES encryption failed." << std::endl; return false; }

        uint32_t crc = calculate_crc32(std::string((const char*)cipher.data(), cipher.size()));

        std::ofstream out(output_path, std::ios::binary);
        if (!out.is_open())
        { MD_LOG_ERROR << "Failed to create encrypted file: " << output_path << std::endl; return false; }

        auto w32 = [&](uint32_t v) { out.write((const char*)&v, 4); };
        out.write(MAGIC.data(), 4); w32(VERSION);
        w32((uint32_t)model_format.size()); out.write(model_format.data(), (std::streamsize)model_format.size());
        out.write((const char*)salt, SALT_LEN); out.write((const char*)iv, IV_LEN);
        w32((uint32_t)cipher.size()); out.write((const char*)cipher.data(), (std::streamsize)cipher.size());
        w32(crc);
        out.close();
        MD_LOG_INFO << "Model encrypted: " << output_path << std::endl;
        return true;
    }

    static bool read_header(const std::string& path, std::string* fmt,
                            std::vector<uint8_t>* salt, std::vector<uint8_t>* iv,
                            std::vector<uint8_t>* cipher, uint32_t* crc_out) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) { MD_LOG_ERROR << "Cannot open: " << path << std::endl; return false; }

        auto r32 = [&](uint32_t* v) { return (bool)in.read((char*)v, 4); };
        char magic[4]; uint32_t ver, fmt_len, data_len;
        if (!in.read(magic, 4) || std::string(magic, 4) != MAGIC)
        { MD_LOG_ERROR << "Bad magic." << std::endl; return false; }
        if (!r32(&ver) || ver != VERSION)
        { MD_LOG_ERROR << "Unsupported version: " << ver << std::endl; return false; }
        if (!r32(&fmt_len)) return false;
        fmt->resize(fmt_len); if (!in.read(&(*fmt)[0], fmt_len)) return false;

        salt->resize(SALT_LEN); iv->resize(IV_LEN);
        if (!in.read((char*)salt->data(), SALT_LEN) || !in.read((char*)iv->data(), IV_LEN)) return false;

        if (!r32(&data_len)) return false;
        cipher->resize(data_len); if (!in.read((char*)cipher->data(), data_len)) return false;
        if (!r32(crc_out)) return false;
        return true;
    }

    bool decrypt_model_file(const std::string& in_path, const std::string& out_path,
                            const std::string& password) {
        if (is_password_invalid(password)) { MD_LOG_ERROR << "Password cannot be empty." << std::endl; return false; }
        std::string fmt; std::vector<uint8_t> salt, iv, cipher; uint32_t crc_ref;
        if (!read_header(in_path, &fmt, &salt, &iv, &cipher, &crc_ref)) return false;

        std::string cs((const char*)cipher.data(), cipher.size());
        if (calculate_crc32(cs) != crc_ref)
        { MD_LOG_ERROR << "CRC mismatch (file corrupted)." << std::endl; return false; }

        uint8_t key[AES_KEY_LEN];
        derive_key(password, salt.data(), (uint32_t)salt.size(), key);

        std::vector<uint8_t> plain;
        if (!aes_decrypt(key, iv.data(), cipher.data(), (uint32_t)cipher.size(), &plain))
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
        std::string fmt; std::vector<uint8_t> s, iv, c; uint32_t crc = 0;
        read_header(path, &fmt, &s, &iv, &c, &crc); return fmt;
    }

    bool read_encrypted_model_to_buffer(const std::string& path, const std::string& password,
                                        std::string* buf, std::string* fmt) {
        if (!buf || !fmt) return false;
        if (is_password_invalid(password)) return false;
        std::vector<uint8_t> salt, iv, cipher; uint32_t crc_ref;
        if (!read_header(path, fmt, &salt, &iv, &cipher, &crc_ref)) return false;
        std::string cs((const char*)cipher.data(), cipher.size());
        if (calculate_crc32(cs) != crc_ref) return false;
        uint8_t key[AES_KEY_LEN];
        derive_key(password, salt.data(), (uint32_t)salt.size(), key);
        std::vector<uint8_t> plain;
        if (!aes_decrypt(key, iv.data(), cipher.data(), (uint32_t)cipher.size(), &plain)) { buf->clear(); fmt->clear(); return false; }
        buf->assign((const char*)plain.data(), plain.size());
        return true;
    }
} // namespace modeldeploy
#endif // ENABLE_ENCRYPTION
