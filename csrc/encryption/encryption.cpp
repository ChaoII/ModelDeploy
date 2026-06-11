//
// Created by aichao on 2025/7/17.
//

#include <fstream>
#include <vector>
#include <cstring>
#include "utils/utils.h"
#include "encryption/encryption.h"

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#else
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#endif

namespace modeldeploy {

    // ==================== SHA-256 密钥派生 ====================
    static bool derive_key(const std::string& password,
                           const uint8_t* salt, uint32_t salt_len,
                           uint8_t* out_key, uint32_t key_len) {
#ifdef _WIN32
        BCRYPT_ALG_HANDLE hAlg = nullptr;
        if (BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) return false;
        BCRYPT_HASH_HANDLE hHash = nullptr;
        if (BCryptCreateHash(hAlg, &hHash, nullptr, 0, nullptr, 0, 0) != 0) {
            BCryptCloseAlgorithmProvider(hAlg, 0); return false;
        }
        // Hash: password || salt
        BCryptHashData(hHash, (PBYTE)password.data(), (ULONG)password.size(), 0);
        BCryptHashData(hHash, (PBYTE)salt, salt_len, 0);
        bool ok = (BCryptFinishHash(hHash, (PBYTE)out_key, key_len, 0) == 0);
        BCryptDestroyHash(hHash);
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return ok;
#else
        SHA256_CTX ctx;
        SHA256_Init(&ctx);
        SHA256_Update(&ctx, password.data(), password.size());
        SHA256_Update(&ctx, salt, salt_len);
        uint8_t full[SHA256_DIGEST_LENGTH];
        SHA256_Final(full, &ctx);
        uint32_t copy = (key_len < SHA256_DIGEST_LENGTH) ? key_len : SHA256_DIGEST_LENGTH;
        memcpy(out_key, full, copy);
        return true;
#endif
    }

    // ==================== AES-256-CBC 加密 ====================
    static bool aes_encrypt(const uint8_t* key, uint32_t key_len,
                            const uint8_t* iv, uint32_t iv_len,
                            const uint8_t* plain, uint32_t plain_len,
                            std::vector<uint8_t>* cipher) {
#ifdef _WIN32
        // BCrypt 会修改 IV 缓冲区，必须拷贝
        uint8_t iv_copy[32];
        memcpy(iv_copy, iv, iv_len);
        BCRYPT_ALG_HANDLE hAlg = nullptr;
        if (BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_AES_ALGORITHM, nullptr, 0) != 0) return false;
        BCryptSetProperty(hAlg, BCRYPT_CHAINING_MODE, (PBYTE)BCRYPT_CHAIN_MODE_CBC,
                          (ULONG)sizeof(BCRYPT_CHAIN_MODE_CBC), 0);
        BCRYPT_KEY_HANDLE hKey = nullptr;
        if (BCryptGenerateSymmetricKey(hAlg, &hKey, nullptr, 0, (PBYTE)key, key_len, 0) != 0) {
            BCryptCloseAlgorithmProvider(hAlg, 0); return false;
        }
        // 计算输出长度（含 PKCS7 padding）
        ULONG out_len = 0;
        BCryptEncrypt(hKey, (PBYTE)plain, plain_len, nullptr,
                      (PBYTE)iv_copy, iv_len, nullptr, 0, &out_len, BCRYPT_BLOCK_PADDING);
        cipher->resize(out_len);
        // 重新拷贝 IV（第一次调用修改了 iv_copy）
        memcpy(iv_copy, iv, iv_len);
        bool ok = (BCryptEncrypt(hKey, (PBYTE)plain, plain_len, nullptr,
                                 (PBYTE)iv_copy, iv_len, cipher->data(), (ULONG)cipher->size(),
                                 &out_len, BCRYPT_BLOCK_PADDING) == 0);
        cipher->resize(out_len);
        BCryptDestroyKey(hKey);
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return ok;
#else
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
#endif
    }

    static bool aes_decrypt(const uint8_t* key, uint32_t key_len,
                            const uint8_t* iv, uint32_t iv_len,
                            const uint8_t* cipher_data, uint32_t cipher_len,
                            std::vector<uint8_t>* plain) {
#ifdef _WIN32
        uint8_t iv_copy[32];
        memcpy(iv_copy, iv, iv_len);
        BCRYPT_ALG_HANDLE hAlg = nullptr;
        if (BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_AES_ALGORITHM, nullptr, 0) != 0) return false;
        BCryptSetProperty(hAlg, BCRYPT_CHAINING_MODE, (PBYTE)BCRYPT_CHAIN_MODE_CBC,
                          (ULONG)sizeof(BCRYPT_CHAIN_MODE_CBC), 0);
        BCRYPT_KEY_HANDLE hKey = nullptr;
        if (BCryptGenerateSymmetricKey(hAlg, &hKey, nullptr, 0, (PBYTE)key, key_len, 0) != 0) {
            BCryptCloseAlgorithmProvider(hAlg, 0); return false;
        }
        ULONG out_len = 0;
        BCryptDecrypt(hKey, (PBYTE)cipher_data, cipher_len, nullptr,
                      (PBYTE)iv_copy, iv_len, nullptr, 0, &out_len, BCRYPT_BLOCK_PADDING);
        plain->resize(out_len);
        memcpy(iv_copy, iv, iv_len);
        bool ok = (BCryptDecrypt(hKey, (PBYTE)cipher_data, cipher_len, nullptr,
                                 (PBYTE)iv_copy, iv_len, plain->data(), (ULONG)plain->size(),
                                 &out_len, BCRYPT_BLOCK_PADDING) == 0);
        plain->resize(out_len);
        BCryptDestroyKey(hKey);
        BCryptCloseAlgorithmProvider(hAlg, 0);
        return ok;
#else
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
#endif
    }

    // ==================== CRC32 ====================
    uint32_t calculate_crc32(const std::string& data) {
        static const uint32_t crc_table[256] = {
            0x00000000L, 0x77073096L, 0xee0e612cL, 0x990951baL, 0x076dc419L,
            0x706af48fL, 0xe963a535L, 0x9e6495a3L, 0x0edb8832L, 0x79dcb8a4L,
            0xe0d5e91eL, 0x97d2d988L, 0x09b64c2bL, 0x7eb17cbdL, 0xe7b82d07L,
            0x90bf1d91L, 0x1db71064L, 0x6ab020f2L, 0xf3b97148L, 0x84be41deL,
            0x1adad47dL, 0x6ddde4ebL, 0xf4d4b551L, 0x83d385c7L, 0x136c9856L,
            0x646ba8c0L, 0xfd62f97aL, 0x8a65c9ecL, 0x14015c4fL, 0x63066cd9L,
            0xfa0f3d63L, 0x8d080df5L,
        };
        uint32_t crc = 0xFFFFFFFF;
        for (const char c : data) {
            crc = crc_table[(crc ^ static_cast<uint8_t>(c)) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFF;
    }

    // ==================== 密码无效检查 ====================
    static bool is_password_invalid(const std::string& password) {
        return password.empty(); // 空密码不安全，禁止使用
    }

    // ==================== 随机字节生成 ====================
    static bool random_bytes(uint8_t* buf, size_t len) {
#ifdef _WIN32
        return BCryptGenRandom(nullptr, buf, (ULONG)len, BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
        return RAND_bytes(buf, (int)len) == 1;
#endif
    }

    // 文件格式（V2）：
    // [4] 魔数 "MDEN"
    // [4] 版本号 (2)
    // [4] 格式字符串长度
    // [N] 格式字符串
    // [16] Salt
    // [16] IV (AES-CBC)
    // [4] 密文长度
    // [N] 密文
    // [4] CRC32

    constexpr static uint32_t SALT_LEN = 16;
    constexpr static uint32_t IV_LEN = 16;
    constexpr static uint32_t AES_KEY_LEN = 32;
    constexpr static uint32_t CURRENT_VERSION = 2;
    const static std::string MAGIC_HEADER = "MDEN";

    bool encrypt_model_file(const std::string& input_path, const std::string& output_path,
                            const std::string& password, const std::string& model_format) {
        if (is_password_invalid(password)) {
            MD_LOG_ERROR << "Password cannot be empty." << std::endl;
            return false;
        }

        std::string model_data;
        if (!read_binary_from_file(input_path, &model_data)) {
            MD_LOG_ERROR << "Failed to read model file: " << input_path << std::endl;
            return false;
        }

        // 生成 salt 和 IV
        uint8_t salt[SALT_LEN], iv[IV_LEN];
        if (!random_bytes(salt, SALT_LEN) || !random_bytes(iv, IV_LEN)) {
            MD_LOG_ERROR << "Failed to generate random bytes." << std::endl;
            return false;
        }

        // 派生密钥
        uint8_t aes_key[AES_KEY_LEN];
        if (!derive_key(password, salt, SALT_LEN, aes_key, AES_KEY_LEN)) {
            MD_LOG_ERROR << "Failed to derive key." << std::endl;
            return false;
        }

        // AES 加密
        std::vector<uint8_t> cipher;
        if (!aes_encrypt(aes_key, AES_KEY_LEN, iv, IV_LEN,
                         reinterpret_cast<const uint8_t*>(model_data.data()),
                         (uint32_t)model_data.size(), &cipher)) {
            MD_LOG_ERROR << "AES encryption failed." << std::endl;
            return false;
        }

        // CRC32（对密文做完整性校验）
        std::string cipher_str(reinterpret_cast<const char*>(cipher.data()), cipher.size());
        uint32_t crc = calculate_crc32(cipher_str);

        // 写入文件
        std::ofstream out(output_path, std::ios::binary);
        if (!out.is_open()) {
            MD_LOG_ERROR << "Failed to create encrypted file: " << output_path << std::endl;
            return false;
        }
        auto write32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
        auto writeN = [&](const void* p, size_t n) { out.write(static_cast<const char*>(p), (std::streamsize)n); };

        writeN(MAGIC_HEADER.data(), 4);
        write32(CURRENT_VERSION);
        write32((uint32_t)model_format.size());
        writeN(model_format.data(), model_format.size());
        writeN(salt, SALT_LEN);
        writeN(iv, IV_LEN);
        write32((uint32_t)cipher.size());
        writeN(cipher.data(), cipher.size());
        write32(crc);

        out.close();
        MD_LOG_INFO << "Model encrypted successfully: " << output_path << std::endl;
        return true;
    }

    // 内部：读取加密文件头并返回各字段偏移
    // 返回 false 表示文件格式错误
    static bool read_encrypted_header(const std::string& file_path,
                                      std::string* model_format,
                                      std::vector<uint8_t>* salt,
                                      std::vector<uint8_t>* iv,
                                      std::vector<uint8_t>* cipher_data,
                                      uint32_t* crc_ref) {
        std::ifstream in(file_path, std::ios::binary);
        if (!in.is_open()) {
            MD_LOG_ERROR << "Failed to open encrypted file: " << file_path << std::endl;
            return false;
        }

        auto read32 = [&](uint32_t* v) -> bool {
            return (bool)in.read(reinterpret_cast<char*>(v), 4);
        };

        char magic[4];
        if (!in.read(magic, 4) || std::string(magic, 4) != MAGIC_HEADER) {
            MD_LOG_ERROR << "Invalid encrypted file format (bad magic)" << std::endl;
            return false;
        }
        uint32_t version;
        if (!read32(&version)) return false;
        if (version != CURRENT_VERSION) {
            MD_LOG_ERROR << "Unsupported version: " << version << std::endl;
            return false;
        }
        uint32_t fmt_len;
        if (!read32(&fmt_len)) return false;
        model_format->resize(fmt_len);
        if (!in.read(&(*model_format)[0], fmt_len)) return false;

        // V2: salt + iv
        if (version >= 2) {
            salt->resize(SALT_LEN);
            iv->resize(IV_LEN);
            if (!in.read(reinterpret_cast<char*>(salt->data()), SALT_LEN)) return false;
            if (!in.read(reinterpret_cast<char*>(iv->data()), IV_LEN)) return false;
        }

        uint32_t data_len;
        if (!read32(&data_len)) return false;
        cipher_data->resize(data_len);
        if (!in.read(reinterpret_cast<char*>(cipher_data->data()), data_len)) return false;

        uint32_t crc;
        if (!read32(&crc)) return false;
        *crc_ref = crc;

        in.close();
        return true;
    }

    bool decrypt_model_file(const std::string& input_path, const std::string& output_path,
                            const std::string& password) {
        if (is_password_invalid(password)) {
            MD_LOG_ERROR << "Password cannot be empty." << std::endl;
            return false;
        }

        std::string model_format;
        std::vector<uint8_t> salt, iv, cipher_data;
        uint32_t crc_ref = 0;

        if (!read_encrypted_header(input_path, &model_format, &salt, &iv, &cipher_data, &crc_ref))
            return false;

        // 校验 CRC
        std::string cs(reinterpret_cast<const char*>(cipher_data.data()), cipher_data.size());
        if (calculate_crc32(cs) != crc_ref) {
            MD_LOG_ERROR << "Decryption failed: CRC mismatch (file corrupted)." << std::endl;
            return false;
        }

        // 派生密钥
        uint8_t aes_key[AES_KEY_LEN];
        if (!derive_key(password, salt.data(), (uint32_t)salt.size(), aes_key, AES_KEY_LEN)) {
            MD_LOG_ERROR << "Failed to derive key." << std::endl;
            return false;
        }

        // AES 解密
        std::vector<uint8_t> plain;
        if (!aes_decrypt(aes_key, AES_KEY_LEN, iv.data(), (uint32_t)iv.size(),
                         cipher_data.data(), (uint32_t)cipher_data.size(), &plain)) {
            MD_LOG_ERROR << "Decryption failed: incorrect password or corrupted file." << std::endl;
            return false;
        }

        // 写入解密后的文件
        std::ofstream out(output_path, std::ios::binary);
        if (!out.is_open()) {
            MD_LOG_ERROR << "Failed to create decrypted file: " << output_path << std::endl;
            return false;
        }
        out.write(reinterpret_cast<const char*>(plain.data()), (std::streamsize)plain.size());
        out.close();
        MD_LOG_INFO << "Model decrypted successfully: " << output_path << std::endl;
        return true;
    }

    bool is_encrypted_model_file(const std::string& file_path) {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) return false;
        char magic[4];
        file.read(magic, 4);
        return std::string(magic, 4) == MAGIC_HEADER;
    }

    std::string get_model_format_from_encrypted_file(const std::string& file_path) {
        std::string fmt;
        std::vector<uint8_t> salt, iv, cipher;
        uint32_t crc = 0;
        read_encrypted_header(file_path, &fmt, &salt, &iv, &cipher, &crc);
        return fmt;
    }

    bool read_encrypted_model_to_buffer(const std::string& file_path, const std::string& password,
                                        std::string* model_buffer, std::string* model_format) {
        if (!model_buffer || !model_format) return false;
        if (is_password_invalid(password)) {
            MD_LOG_ERROR << "Password cannot be empty." << std::endl;
            return false;
        }

        std::vector<uint8_t> salt, iv, cipher_data;
        uint32_t crc_ref = 0;
        if (!read_encrypted_header(file_path, model_format, &salt, &iv, &cipher_data, &crc_ref))
            return false;

        std::string cs(reinterpret_cast<const char*>(cipher_data.data()), cipher_data.size());
        if (calculate_crc32(cs) != crc_ref) {
            MD_LOG_ERROR << "Decryption failed: CRC mismatch (file corrupted)." << std::endl;
            return false;
        }

        uint8_t aes_key[AES_KEY_LEN];
        if (!derive_key(password, salt.data(), (uint32_t)salt.size(), aes_key, AES_KEY_LEN)) {
            MD_LOG_ERROR << "Failed to derive key." << std::endl;
            return false;
        }

        std::vector<uint8_t> plain;
        if (!aes_decrypt(aes_key, AES_KEY_LEN, iv.data(), (uint32_t)iv.size(),
                         cipher_data.data(), (uint32_t)cipher_data.size(), &plain)) {
            MD_LOG_ERROR << "Decryption failed: incorrect password or corrupted file." << std::endl;
            model_buffer->clear();
            model_format->clear();
            return false;
        }

        model_buffer->assign(reinterpret_cast<const char*>(plain.data()), plain.size());
        return true;
    }

} // namespace modeldeploy
