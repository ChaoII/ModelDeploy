//
// Created by aichao on 2025/7/17.
//

#include "encryption/encryption.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout <<
            "usage: model_encrypted encrypt/decrypt input_model_path output_model_path password [format(mnn onnx engine)]\n";
        return 1;
    }
    std::string mode = argv[1];
    std::string in_path = argv[2];
    std::string out_path = argv[3];
    std::string password = argv[4];
    if (mode == "encrypt") {
        if (argc < 6) {
            std::cout << "encrypt must a model format(onnx/mnn/engine)\n";
            return 1;
        }
        const std::string format = argv[5];
        if (modeldeploy::encrypt_model_file(in_path, out_path, password, format))
            std::cout << "encrypt success" << std::endl;
        else
            std::cout << "encrypt success" << std::endl;
    }
    else if (mode == "decrypt") {
        if (modeldeploy::decrypt_model_file(in_path, out_path, password))
            std::cout << "decrypt success" << std::endl;
        else
            std::cout << "decrypt success" << std::endl;
    }
    else {
        std::cout << "unknown mode must one of [encrypt, decrypt] \n";
    }
    return 0;
}
