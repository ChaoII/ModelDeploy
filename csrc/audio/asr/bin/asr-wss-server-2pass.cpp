
#include "websocket-server-2pass.h"

#ifdef _WIN32

#include "win_func.h"

#else
#include <unistd.h>
#endif

#include <fstream>
#include "util.h"
#include "src/utils/internal/utils.h"

// hot_words
std::unordered_map<std::string, int> hws_map_;
int fst_inc_wts_ = 20;
float global_beam_, lattice_beam_, am_scale_;


int main(int argc, char *argv[]) {
#ifdef _WIN32

#include <windows.h>

    SetConsoleOutputCP(65001);
#endif
    try {
        google::InitGoogleLogging(argv[0]);
        FLAGS_logtostderr = true;
        TCLAP::CmdLine cmd("asr-wss-server", ' ', "");

        TCLAP::ValueArg<std::string> offline_model_dir(
                "", OFFLINE_MODEL_DIR,
                " the asr model path, which contains model_quant.onnx, config.yaml, am.mvn",
                false, "", "string");
        TCLAP::ValueArg<std::string> online_model_dir(
                "", ONLINE_MODEL_DIR,
                "the asr model path, which contains model_quant.onnx, config.yaml, am.mvn",
                false, "", "string");

        TCLAP::ValueArg<std::string> vad_dir(
                "", VAD_DIR,
                "the vad model path, which contains model_quant.onnx, vad.yaml, vad.mvn",
                false, "", "string");

        TCLAP::ValueArg<std::string> punct_dir(
                "", PUNC_DIR,
                "the punc model path, which contains model_quant.onnx, punc.yaml",
                false, "", "string");

        TCLAP::ValueArg<std::string> itn_dir(
                "", ITN_DIR,
                "the itn model path, which contains zh_itn_tagger.fst, zh_itn_verbalizer.fst",
                false, "", "string");

        TCLAP::ValueArg<std::string> listen_ip(
                "", "listen-ip",
                "listen ip",
                false, "0.0.0.0", "string");
        TCLAP::ValueArg<int> port(
                "", "port",
                "port",
                false, 10095, "int");
        TCLAP::ValueArg<int> io_thread_num(
                "", "io-thread-num",
                "io thread num",
                false, 2, "int");
        TCLAP::ValueArg<int> decoder_thread_num(
                "", "decoder-thread-num",
                "decoder thread num",
                false, 8, "int");
        TCLAP::ValueArg<int> model_thread_num(
                "", "model-thread-num",
                "model thread num",
                false, 2, "int");

        TCLAP::ValueArg<std::string> cert_file(
                "", "cert-file",
                "path of certificate for WSS connection. if it is empty, it will be in WS mode.",
                false, "./ssl_key/server.crt", "string");
        TCLAP::ValueArg<std::string> key_file(
                "", "key-file",
                "path of key_file for WSS connection",
                false, "./ssl_key/server.key", "string");

        TCLAP::ValueArg<float> global_beam(
                "", GLOB_BEAM,
                "the decoding beam for beam searching ",
                false, 3.0, "float");
        TCLAP::ValueArg<float> lattice_beam(
                "", LAT_BEAM,
                "the lattice generation beam for beam searching ",
                false, 3.0, "float");
        TCLAP::ValueArg<float> am_scale(
                "", AM_SCALE,
                "the acoustic scale for beam searching ",
                false, 10.0, "float");

        TCLAP::ValueArg<std::string> lm_dir(
                "", LM_DIR,
                "the LM model path, which contains compiled models: TLG.fst, config.yaml ",
                false, "", "string");

        TCLAP::ValueArg<std::string> hot_word(
                "", HOTWORD,
                "the hot_word file, one hot_word per-line, Format: Hot_word Weight (could be: 阿里巴巴 20)",
                false, "", "string");
        TCLAP::ValueArg<std::int32_t> fst_inc_wts(
                "", FST_INC_WTS,
                "the fst hot_words incremental bias",
                false, 20, "int32_t");


        cmd.add(hot_word);
        cmd.add(fst_inc_wts);
        cmd.add(global_beam);
        cmd.add(lattice_beam);
        cmd.add(am_scale);

        cmd.add(cert_file);
        cmd.add(key_file);

        cmd.add(offline_model_dir);
        cmd.add(online_model_dir);
        cmd.add(vad_dir);
        cmd.add(punct_dir);

        cmd.add(itn_dir);
        cmd.add(lm_dir);

        cmd.add(listen_ip);
        cmd.add(port);
        cmd.add(io_thread_num);
        cmd.add(decoder_thread_num);
        cmd.add(model_thread_num);
        cmd.parse(argc, argv);


        std::map<std::string, std::string> model_path;
        model_path.insert({OFFLINE_MODEL_DIR, offline_model_dir.getValue()});
        model_path.insert({ONLINE_MODEL_DIR, online_model_dir.getValue()});
        model_path.insert({QUANTIZE, is_quantize_model(offline_model_dir.getValue()) ? "true" : "false"});
        model_path.insert({VAD_DIR, vad_dir.getValue()});
        model_path.insert({VAD_QUANT, is_quantize_model(vad_dir.getValue()) ? "true" : "false"});
        model_path.insert({PUNC_DIR, punct_dir.getValue()});
        model_path.insert({PUNC_QUANT, is_quantize_model(punct_dir.getValue()) ? "true" : "false"});
        model_path.insert({ITN_DIR, itn_dir.getValue()});
        model_path.insert({LM_DIR, lm_dir.getValue()});
        model_path.insert({HOTWORD, hot_word.getValue()});


        global_beam_ = global_beam.getValue();
        lattice_beam_ = lattice_beam.getValue();
        am_scale_ = am_scale.getValue();

        try {
            std::string s_vad_path = model_path[VAD_DIR];
            std::string s_vad_quant = model_path[VAD_QUANT];
            std::string s_offline_asr_path = model_path[OFFLINE_MODEL_DIR];
            std::string s_online_asr_path = model_path[ONLINE_MODEL_DIR];
            std::string s_asr_quant = model_path[QUANTIZE];
            std::string s_punc_path = model_path[PUNC_DIR];
            std::string s_punc_quant = model_path[PUNC_QUANT];
            std::string s_itn_path = model_path[ITN_DIR];
            std::string s_lm_path = model_path[LM_DIR];

            if (s_vad_path.empty()) {
                LOG(WARNING) << "VAD model is not set, use default.";
            }
            if (s_offline_asr_path.empty()) {
                LOG(ERROR) << "offline asr model path must be specified";
            }

            if (s_online_asr_path.empty()) {
                LOG(ERROR) << "online asr model path must be specified";
            }

            if (s_lm_path.empty()) {
                LOG(ERROR) << "LM model is not set, not executed.";
            }

            if (s_punc_path.empty()) {
                LOG(ERROR) << "punct model is not set, use default.";
            }

            if (s_itn_path.empty()) {
                LOG(ERROR) << "itn path is not set";
            }

        } catch (std::exception const &e) {
            LOG(ERROR) << "Error: " << e.what();
        }

        std::string s_listen_ip = listen_ip.getValue();
        int s_port = port.getValue();
        int s_io_thread_num = io_thread_num.getValue();
        int s_decoder_thread_num = decoder_thread_num.getValue();

        int s_model_thread_num = model_thread_num.getValue();

        asio::io_context io_decoder;  // context for decoding
        asio::io_context io_server;   // context for server

        std::vector<std::thread> decoder_threads;

        std::string s_cert_file = cert_file.getValue();
        std::string s_keyfile = key_file.getValue();

        // hot_word file
        std::string hot_word_path;
        hot_word_path = model_path.at(HOTWORD);
        fst_inc_wts_ = fst_inc_wts.getValue();
        LOG(INFO) << "hot_word path: " << hot_word_path;
        ExtractHotWords(hot_word_path, hws_map_);

        bool is_ssl = false;
        if (!s_cert_file.empty() && access(s_cert_file.c_str(), F_OK) == 0) {
            is_ssl = true;
        }

        auto conn_guard = asio::make_work_guard(io_decoder);  // make sure threads can wait in the queue
        auto server_guard = asio::make_work_guard(io_server);  // make sure threads can wait in the queue
        // create threads pool
        decoder_threads.reserve(s_decoder_thread_num);
        for (int32_t i = 0; i < s_decoder_thread_num; ++i) {
            decoder_threads.emplace_back([&io_decoder]() { io_decoder.run(); });
        }

        server server_;  // server for websocket
        wss_server wss_server_;
        server *server = nullptr;
        wss_server *wss_server = nullptr;
        if (is_ssl) {
            LOG(INFO) << "SSL is opened!";
            wss_server_.init_asio(&io_server);  // init asio
            wss_server_.set_reuse_addr(true);  // reuse address as we create multiple threads
            // list on port for accept
            wss_server_.listen(asio::ip::address::from_string(s_listen_ip), s_port);
            wss_server = &wss_server_;
        } else {
            LOG(INFO) << "SSL is closed!";
            server_.init_asio(&io_server);  // init asio
            server_.set_reuse_addr(true);  // reuse address as we create multiple threads
            // list on port for accept
            server_.listen(asio::ip::address::from_string(s_listen_ip), s_port);
            server = &server_;
        }

        WebSocketServer websocket_srv(io_decoder, is_ssl, server, wss_server, s_cert_file,
                                      s_keyfile);  // websocket server for asr engine
        websocket_srv.initAsr(model_path, s_model_thread_num);  // init asr model

        LOG(INFO) << "decoder-thread-num: " << s_decoder_thread_num;
        LOG(INFO) << "io-thread-num: " << s_io_thread_num;
        LOG(INFO) << "model-thread-num: " << s_model_thread_num;
        LOG(INFO) << "asr model init finished. listen on port:" << s_port;

        // Start the ASIO network io_service run loop
        std::vector<std::thread> ts;
        // create threads for io network
        for (size_t i = 0; i < s_io_thread_num; i++) {
            ts.emplace_back([&io_server]() { io_server.run(); });
        }
        // wait for threads
        for (size_t i = 0; i < s_io_thread_num; i++) {
            ts[i].join();
        }

        // wait for threads
        for (auto &t: decoder_threads) {
            t.join();
        }

    } catch (std::exception const &e) {
        LOG(ERROR) << e.what();
    }
    return 0;
}
