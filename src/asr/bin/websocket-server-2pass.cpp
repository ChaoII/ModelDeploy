
#include <thread>
#include <utility>
#include <vector>
#include "websocket-server-2pass.h"


extern std::unordered_map<std::string, int> hws_map_;
extern int fst_inc_wts_;
extern float global_beam_, lattice_beam_, am_scale_;

context_ptr WebSocketServer::on_tls_init(tls_mode mode,
                                         const websocketpp::connection_hdl &hdl,
                                         std::string &s_cert_file,
                                         std::string &s_key_file) {
    namespace asio = websocketpp::lib::asio;

    LOG(INFO) << "on_tls_init called with hdl: " << hdl.lock().get();
    LOG(INFO) << "using TLS mode: "
              << (mode == MOZILLA_MODERN ? "Mozilla Modern"
                                         : "Mozilla Intermediate");
    context_ptr ctx = websocketpp::lib::make_shared<asio::ssl::context>(
            asio::ssl::context::sslv23);

    try {
        if (mode == MOZILLA_MODERN) {
            // Modern disables TLSv1
            ctx->set_options(
                    asio::ssl::context::default_workarounds |
                    asio::ssl::context::no_sslv2 | asio::ssl::context::no_sslv3 |
                    asio::ssl::context::no_tlsv1 | asio::ssl::context::single_dh_use);
        } else {
            ctx->set_options(asio::ssl::context::default_workarounds |
                             asio::ssl::context::no_sslv2 |
                             asio::ssl::context::no_sslv3 |
                             asio::ssl::context::single_dh_use);
        }
        ctx->use_certificate_chain_file(s_cert_file);
        ctx->use_private_key_file(s_key_file, asio::ssl::context::pem);
    } catch (std::exception &e) {
        LOG(ERROR) << "Exception: " << e.what();
    }
    return ctx;
}

nlohmann::json handle_result(FUNASR_RESULT result) {
    websocketpp::lib::error_code();
    nlohmann::json json_result;
    json_result["text"] = "";

    std::string tmp_online_msg = FunASRGetResult(result, 0);
    if (!tmp_online_msg.empty()) {
        LOG(INFO) << "online_res :" << tmp_online_msg;
        json_result["text"] = tmp_online_msg;
        json_result["mode"] = "2pass-online";
    }
    std::string tmp_two_pass_msg = FunASRGetTpassResult(result, 0);
    if (!tmp_two_pass_msg.empty()) {
        LOG(INFO) << "offline results : " << tmp_two_pass_msg;
        json_result["text"] = tmp_two_pass_msg;
        json_result["mode"] = "2pass-offline";
    }

    std::string tmp_stamp_msg = FunASRGetStamp(result);
    if (!tmp_stamp_msg.empty()) {
        LOG(INFO) << "offline stamps : " << tmp_stamp_msg;
        json_result["timestamp"] = tmp_stamp_msg;
    }

    std::string tmp_stamp_sentences = FunASRGetStampSents(result);
    if (!tmp_stamp_sentences.empty()) {
        try {
            nlohmann::json json_stamp = nlohmann::json::parse(tmp_stamp_sentences);
            LOG(INFO) << "offline stamp_sentences : " << json_stamp;
            json_result["stamp_sentences"] = json_stamp;
        } catch (std::exception const &e) {
            LOG(ERROR) << tmp_stamp_sentences << e.what();
            json_result["stamp_sentences"] = "";
        }
    }

    return json_result;
}

// feed buffer to asr engine for decoder
void WebSocketServer::do_decoder(
        std::vector<char> &buffer,
        websocketpp::connection_hdl &hdl,
        nlohmann::json &msg,
        std::vector<std::vector<std::string>> &punct_cache,
        std::vector<std::vector<float>> &hot_words_embedding,
        websocketpp::lib::mutex &thread_lock,
        bool &is_final,
        const std::string &wav_name,
        const std::string &mode_type,
        bool itn,
        int audio_fs,
        const std::string &wav_format,
        FUNASR_HANDLE &two_pass_online_handle,
        FUNASR_DEC_HANDLE &decoder_handle,
        const std::string &svs_lang,
        bool sys_itn) {
    // lock for each connection
    if (!two_pass_online_handle) {
        scoped_lock guard(thread_lock);
        LOG(INFO) << "two_pass_online_handle is free, return";
        msg["access_num"] = (int) msg["access_num"] - 1;
        return;
    }
    try {
        FUNASR_RESULT Result;
        int asr_mode_ = 2;
        if (mode_type == "offline") {
            asr_mode_ = 0;
        } else if (mode_type == "online") {
            asr_mode_ = 1;
        } else if (mode_type == "2pass") {
            asr_mode_ = 2;
        }
        while (buffer.size() >= 800 * 2 && !msg["is_eof"]) {
            std::vector<char> sub_vector = {buffer.begin(), buffer.begin() + 800 * 2};
            buffer.erase(buffer.begin(), buffer.begin() + 800 * 2);
            try {
                if (two_pass_online_handle) {
                    Result = FunTpassInferBuffer(two_pass_handle, two_pass_online_handle,
                                                 sub_vector.data(),
                                                 static_cast<int>(sub_vector.size()),
                                                 punct_cache, false, audio_fs,
                                                 wav_format, (ASR_TYPE) asr_mode_,
                                                 hot_words_embedding, itn, decoder_handle,
                                                 svs_lang, sys_itn);
                } else {
                    scoped_lock guard(thread_lock);
                    msg["access_num"] = (int) msg["access_num"] - 1;
                    return;
                }
            } catch (std::exception const &e) {
                scoped_lock guard(thread_lock);
                LOG(ERROR) << e.what();
                msg["access_num"] = (int) msg["access_num"] - 1;
                return;
            }
            if (Result) {
                websocketpp::lib::error_code ec;
                nlohmann::json json_result = handle_result(Result);
                json_result["wav_name"] = wav_name;
                json_result["is_final"] = false;
                if (!json_result["text"].empty()) {
                    if (is_ssl) {
                        wss_server_->send(hdl, json_result.dump(),
                                          websocketpp::frame::opcode::text, ec);
                    } else {
                        server_->send(hdl, json_result.dump(),
                                      websocketpp::frame::opcode::text, ec);
                    }
                }
                FunASRFreeResult(Result);
            }
        }
        if (is_final && !msg["is_eof"]) {
            try {
                if (two_pass_online_handle) {
                    Result = FunTpassInferBuffer(two_pass_handle, two_pass_online_handle,
                                                 buffer.data(),
                                                 static_cast<int>(buffer.size()),
                                                 punct_cache,
                                                 is_final, audio_fs,
                                                 wav_format, (ASR_TYPE) asr_mode_,
                                                 hot_words_embedding, itn, decoder_handle,
                                                 svs_lang, sys_itn);
                } else {
                    scoped_lock guard(thread_lock);
                    msg["access_num"] = (int) msg["access_num"] - 1;
                    return;
                }
            } catch (std::exception const &e) {
                scoped_lock guard(thread_lock);
                LOG(ERROR) << e.what();
                msg["access_num"] = (int) msg["access_num"] - 1;
                return;
            }
            if (!punct_cache.empty()) {
                for (auto &vec: punct_cache) {
                    vec.clear();
                }
            }
            if (Result) {
                websocketpp::lib::error_code ec;
                nlohmann::json json_result = handle_result(Result);
                json_result["wav_name"] = wav_name;
                json_result["is_final"] = true;
                if (is_ssl) {
                    wss_server_->send(hdl, json_result.dump(), websocketpp::frame::opcode::text, ec);
                } else {
                    server_->send(hdl, json_result.dump(), websocketpp::frame::opcode::text, ec);
                }
                FunASRFreeResult(Result);
            } else {
                if (wav_format != "pcm" && wav_format != "PCM") {
                    websocketpp::lib::error_code ec;
                    nlohmann::json json_result;
                    json_result["text"] = "ERROR. Real-time transcription service ONLY SUPPORT PCM stream.";
                    json_result["wav_name"] = wav_name;
                    json_result["is_final"] = true;
                    if (is_ssl) {
                        wss_server_->send(hdl, json_result.dump(), websocketpp::frame::opcode::text, ec);
                    } else {
                        server_->send(hdl, json_result.dump(), websocketpp::frame::opcode::text, ec);
                    }
                }
            }
        }
    } catch (std::exception const &e) {
        LOG(ERROR) << e.what();
    }
    scoped_lock guard(thread_lock);
    msg["access_num"] = (int) msg["access_num"] - 1;
}

void WebSocketServer::on_open(const websocketpp::connection_hdl &hdl) {
    scoped_lock guard(m_lock);     // for threads safety
    try {
        // put a new data vector for new
        std::shared_ptr<ASR_MESSAGE> data_msg = std::make_shared<ASR_MESSAGE>();
        // connection
        data_msg->samples = std::make_shared<std::vector<char>>();
        data_msg->thread_lock = std::make_shared<websocketpp::lib::mutex>();
        data_msg->msg = nlohmann::json::parse("{}");
        data_msg->msg["wav_format"] = "pcm";
        data_msg->msg["wav_name"] = "wav-default-id";
        data_msg->msg["mode"] = "2pass";
        data_msg->msg["itn"] = true;
        data_msg->msg["audio_fs"] = 16000; // default is 16k
        data_msg->msg["access_num"] = 0; // the number of access for this object, when it is 0, we can free it safely
        data_msg->msg["is_eof"] = false; // if this connection is closed
        data_msg->msg["svs_lang"] = "auto";
        data_msg->msg["svs_itn"] = true;
        FUNASR_DEC_HANDLE decoder_handle =
                FunASRWfstDecoderInit(two_pass_handle, ASR_TWO_PASS, global_beam_, lattice_beam_, am_scale_);
        data_msg->decoder_handle = decoder_handle;
        data_msg->punct_cache = std::make_shared<std::vector<std::vector<std::string>>>(2);
        data_msg->strand_ = std::make_shared<asio::io_context::strand>(io_decoder_);

        data_map.emplace(hdl, data_msg);
    } catch (std::exception const &e) {
        LOG(ERROR) << e.what();
    }
}

void remove_hdl(const websocketpp::connection_hdl &hdl, WSS_Data_Map &data_map) {
    std::shared_ptr<ASR_MESSAGE> data_msg;
    auto it_data = data_map.find(hdl);
    if (it_data != data_map.end()) {
        data_msg = it_data->second;
    } else {
        return;
    }
    // finished and avoid access freed two_pass_online_handle
    unique_lock guard_decoder(*(data_msg->thread_lock));
    if (data_msg->msg["access_num"] == 0 && data_msg->msg["is_eof"] == true) {
        FunWfstDecoderUnloadHwsRes(data_msg->decoder_handle);
        FunASRWfstDecoderUninit(data_msg->decoder_handle);
        data_msg->decoder_handle = nullptr;
        FunTpassOnlineUninit(data_msg->two_pass_online_handle);
        data_msg->two_pass_online_handle = nullptr;
        data_map.erase(hdl);
    }
    guard_decoder.unlock();
}

void WebSocketServer::on_close(const websocketpp::connection_hdl &hdl) {
    scoped_lock guard(m_lock);
    std::shared_ptr<ASR_MESSAGE> data_msg;
    auto it_data = data_map.find(hdl);
    if (it_data != data_map.end()) {
        data_msg = it_data->second;
    } else {
        return;
    }
    unique_lock guard_decoder(*(data_msg->thread_lock));
    data_msg->msg["is_eof"] = true;
    guard_decoder.unlock();
}

// remove closed connection
void WebSocketServer::check_and_clean_connection() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        std::vector<websocketpp::connection_hdl> to_remove;  // remove list
        auto iter = data_map.begin();
        while (iter != data_map.end()) {  // loop to find closed connection
            websocketpp::connection_hdl hdl = iter->first;
            try {
                if (is_ssl) {
                    wss_server::connection_ptr con = wss_server_->get_con_from_hdl(hdl);
                    if (con->get_state() != 1) {  // session::state::open ==1
                        to_remove.push_back(hdl);
                    }
                } else {
                    server::connection_ptr con = server_->get_con_from_hdl(hdl);
                    if (con->get_state() != 1) {  // session::state::open ==1
                        to_remove.push_back(hdl);
                    }
                }
            }
            catch (std::exception const &e) {
                // if connection is close, we set is_eof = true
                std::shared_ptr<ASR_MESSAGE> data_msg;
                auto it_data = data_map.find(hdl);
                if (it_data != data_map.end()) {
                    data_msg = it_data->second;
                } else {
                    continue;
                }
                unique_lock guard_decoder(*(data_msg->thread_lock));
                data_msg->msg["is_eof"] = true;
                guard_decoder.unlock();
                to_remove.push_back(hdl);
                LOG(INFO) << "connection is closed.";
            }
            iter++;
        }
        for (const auto &hdl: to_remove) {
            {
                unique_lock lock(m_lock);
                remove_hdl(hdl, data_map);
            }
        }
    }
}

void WebSocketServer::on_message(websocketpp::connection_hdl hdl, const message_ptr &msg) {
    unique_lock lock(m_lock);
    // find the sample data vector according to one connection
    std::shared_ptr<ASR_MESSAGE> msg_data;
    auto it_data = data_map.find(hdl);
    if (it_data != data_map.end()) {
        msg_data = it_data->second;
        if (msg_data->msg["is_eof"]) {
            lock.unlock();
            return;
        }
    } else {
        lock.unlock();
        return;
    }
    auto sample_data_p = msg_data->samples;
    auto punct_cache_p = msg_data->punct_cache;
    auto thread_lock_p = msg_data->thread_lock;
    lock.unlock();

    if (sample_data_p == nullptr) {
        LOG(INFO) << "error when fetch sample data vector";
        return;
    }

    const std::string &payload = msg->get_payload();  // get msg type
    unique_lock guard_decoder(*(thread_lock_p)); // mutex for one connection
    switch (msg->get_opcode()) {
        case websocketpp::frame::opcode::text: {
            nlohmann::json json_result;
            try {
                json_result = nlohmann::json::parse(payload);
            } catch (std::exception const &e) {
                LOG(ERROR) << e.what();
                msg_data->msg["is_eof"] = true;
                guard_decoder.unlock();
                return;
            }

            if (json_result.contains("wav_name")) {
                msg_data->msg["wav_name"] = json_result["wav_name"];
            }
            if (json_result.contains("mode")) {
                msg_data->msg["mode"] = json_result["mode"];
            }
            if (json_result.contains("wav_format")) {
                msg_data->msg["wav_format"] = json_result["wav_format"];
            }

            // hot_words: fst/nn
            if (msg_data->hot_words_embedding == nullptr) {
                std::unordered_map<std::string, int> merged_hws_map;
                std::string nn_hot_words;

                if (json_result["hot_words"] != nullptr) {
                    std::string json_string = json_result["hot_words"];
                    if (!json_string.empty()) {
                        nlohmann::json json_fst_hws;
                        try {
                            json_fst_hws = nlohmann::json::parse(json_string);
                            if (json_fst_hws.type() == nlohmann::json::value_t::object) {
                                // fst
                                try {
                                    std::unordered_map<std::string, int> client_hws_map = json_fst_hws;
                                    merged_hws_map.insert(client_hws_map.begin(), client_hws_map.end());
                                } catch (const std::exception &e) {
                                    LOG(INFO) << e.what();
                                }
                            }
                        } catch (std::exception const &e) {
                            LOG(ERROR) << e.what();
                            // nn
                            std::string client_nn_hws = json_result["hot_words"];
                            nn_hot_words += " " + client_nn_hws;
                            LOG(INFO) << "nn hot_words: " << client_nn_hws;
                        }
                    }
                }
                merged_hws_map.insert(hws_map_.begin(), hws_map_.end());

                // fst
                LOG(INFO) << "hot_words: ";
                for (const auto &pair: merged_hws_map) {
                    nn_hot_words += " " + pair.first;
                    LOG(INFO) << pair.first << " : " << pair.second;
                }
                FunWfstDecoderLoadHwsRes(msg_data->decoder_handle, fst_inc_wts_, merged_hws_map);

                // nn
                auto new_hot_words_embedding = CompileHotwordEmbedding(two_pass_handle, nn_hot_words, ASR_TWO_PASS);
                msg_data->hot_words_embedding =
                        std::make_shared<std::vector<std::vector<float>>>(new_hot_words_embedding);
            }

            if (json_result.contains("audio_fs")) {
                msg_data->msg["audio_fs"] = json_result["audio_fs"];
            }
            if (json_result.contains("chunk_size")) {
                if (msg_data->two_pass_online_handle == nullptr) {
                    std::vector<int> chunk_size_vec =
                            json_result["chunk_size"].get<std::vector<int>>();
                    // check chunk_size_vec
                    if (chunk_size_vec.size() == 3 && chunk_size_vec[1] != 0) {
                        FUNASR_HANDLE two_pass_online_handle =
                                FunTpassOnlineInit(two_pass_handle, chunk_size_vec);
                        msg_data->two_pass_online_handle = two_pass_online_handle;
                    } else {
                        LOG(ERROR) << "Wrong chunk_size!";
                        break;
                    }
                }
            }
            if (json_result.contains("itn")) {
                msg_data->msg["itn"] = json_result["itn"];
            }
            if (json_result.contains("svs_lang")) {
                msg_data->msg["svs_lang"] = json_result["svs_lang"];
            }
            if (json_result.contains("svs_itn")) {
                msg_data->msg["svs_itn"] = json_result["svs_itn"];
            }
            LOG(INFO) << "json_result=" << json_result << ", msg_data->msg=" << msg_data->msg;
            if ((json_result["is_speaking"] == false ||
                 json_result["is_finished"] == true) &&
                msg_data->msg["is_eof"] != true &&
                msg_data->hot_words_embedding != nullptr) {
                LOG(INFO) << "client done";
                // if it is in final message, post the sample_data to decode
                try {
                    std::vector<std::vector<float>> hot_words_embedding_(*(msg_data->hot_words_embedding));
                    msg_data->strand_->post(
                            std::bind(&WebSocketServer::do_decoder, this,
                                      std::move(*(sample_data_p)), std::move(hdl),
                                      std::ref(msg_data->msg), std::ref(*punct_cache_p),
                                      std::move(hot_words_embedding_),
                                      std::ref(*thread_lock_p), true,
                                      msg_data->msg["wav_name"],
                                      msg_data->msg["mode"],
                                      msg_data->msg["itn"],
                                      msg_data->msg["audio_fs"],
                                      msg_data->msg["wav_format"],
                                      std::ref(msg_data->two_pass_online_handle),
                                      std::ref(msg_data->decoder_handle),
                                      msg_data->msg["svs_lang"],
                                      msg_data->msg["svs_itn"]));
                    msg_data->msg["access_num"] = (int) (msg_data->msg["access_num"]) + 1;
                }
                catch (std::exception const &e) {
                    LOG(ERROR) << e.what();
                }
            }
            break;
        }
        case websocketpp::frame::opcode::binary: {
            // received binary data
            const auto *pcm_data = static_cast<const char *>(payload.data());
            auto num_samples = static_cast<int32_t>(payload.size());
            if (is_online) {
                sample_data_p->insert(sample_data_p->end(), pcm_data, pcm_data + num_samples);
                int set_p_size = 800 * 2;  // TODO, need get from client
                // if sample_data size > set_p_size, we post data to decode
                if (sample_data_p->size() > set_p_size) {
                    int chunk_size = floor(sample_data_p->size() / set_p_size);
                    // make sure the sub_vector size is an integer multiple of setpsize
                    std::vector<char> sub_vector = {sample_data_p->begin(),
                                                    sample_data_p->begin() + chunk_size * set_p_size};
                    // keep remain in sample_data
                    sample_data_p->erase(sample_data_p->begin(),
                                         sample_data_p->begin() + chunk_size * set_p_size);
                    try {
                        // post to decode
                        if (msg_data->msg["is_eof"] != true && msg_data->hot_words_embedding != nullptr) {
                            std::vector<std::vector<float>> hot_words_embedding_(*(msg_data->hot_words_embedding));
                            msg_data->strand_->post(
                                    std::bind(&WebSocketServer::do_decoder, this,
                                              std::move(sub_vector), std::move(hdl),
                                              std::ref(msg_data->msg),
                                              std::ref(*punct_cache_p),
                                              std::move(hot_words_embedding_),
                                              std::ref(*thread_lock_p), false,
                                              msg_data->msg["wav_name"],
                                              msg_data->msg["mode"],
                                              msg_data->msg["itn"],
                                              msg_data->msg["audio_fs"],
                                              msg_data->msg["wav_format"],
                                              std::ref(msg_data->two_pass_online_handle),
                                              std::ref(msg_data->decoder_handle),
                                              msg_data->msg["svs_lang"],
                                              msg_data->msg["svs_itn"]));
                            msg_data->msg["access_num"] = (int) (msg_data->msg["access_num"]) + 1;
                        }
                    }
                    catch (std::exception const &e) {
                        LOG(ERROR) << e.what();
                    }
                }
            } else {
                sample_data_p->insert(sample_data_p->end(), pcm_data, pcm_data + num_samples);
            }
            break;
        }
        default:
            break;
    }
    guard_decoder.unlock();
}

// init asr model
void WebSocketServer::initAsr(std::map<std::string, std::string> &model_path, int thread_num) {
    try {
        two_pass_handle = FunTpassInit(model_path, thread_num);
        if (!two_pass_handle) {
            LOG(ERROR) << "FunTpassInit init failed";
            exit(-1);
        }
        LOG(INFO) << "initAsr run check_and_clean_connection";
        std::thread clean_thread(&WebSocketServer::check_and_clean_connection, this);
        clean_thread.detach();
        LOG(INFO) << "initAsr run check_and_clean_connection finished";
    } catch (const std::exception &e) {
        LOG(INFO) << e.what();
    }
}
