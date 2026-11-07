// Private persistent worker for the host. ASCII request header + exact UTF-8 byte
// count on stdin; bounded NDJSON events on stdout. No user text is a command.
#pragma once
#include <csignal>
#include <iostream>
#include <sstream>

namespace lfm2 {
inline volatile std::sig_atomic_t worker_cancelled = 0;
inline void worker_cancel(int) { worker_cancelled = 1; }
inline std::string worker_hex(const std::string& s) {
    static const char* digits = "0123456789abcdef";
    std::string out; out.reserve(s.size() * 2);
    for (unsigned char c : s) { out += digits[c >> 4]; out += digits[c & 15]; }
    return out;
}
inline void worker_error(const std::string& message) {
    std::printf("{\"type\":\"error\",\"hex\":\"%s\"}\n", worker_hex(message).c_str());
    std::fflush(stdout);
}

template<class E>
int run_worker(E& en, const Tokenizer& tokenizer, const Detok& dt, const Options& base) {
    struct sigaction sa{}; sa.sa_handler = worker_cancel; sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &sa, nullptr);
    std::puts("{\"type\":\"ready\",\"protocol\":1}"); std::fflush(stdout);
    std::mt19937 rng(std::random_device{}());
    std::vector<float> logits(en.V);
    std::string header;
    while (std::getline(std::cin, header)) {
        Options o = base;
        std::string magic, extra;
        size_t bytes = 0;
        std::istringstream in(header);
        if (header.size() > 256 || !(in >> magic >> o.n_gen >> o.top_k >> o.temp >> bytes) ||
            (in >> extra) || magic != "LFM2/1" || bytes > 4 * 1024 * 1024 ||
            o.n_gen < 0 || o.n_gen > o.ctx || o.top_k < 0 || o.top_k > en.V ||
            !std::isfinite(o.temp) || o.temp < 0 || o.temp > 2) {
            worker_error("invalid worker request"); return 1;
        }
        o.text.assign(bytes, '\0');
        if (!std::cin.read(o.text.data(), bytes)) return 1;
        o.prompt = true; o.chat = false;
        worker_cancelled = 0;
        try {
            const auto ids = prompt_ids(o, tokenizer);
            // Zero means no output cap: use all remaining positions in context.
            if (o.n_gen == 0) o.n_gen = o.ctx - (int)ids.size();
            if (o.n_gen < 1) throw std::runtime_error("Contexto cheio. Inicie uma nova conversa ou reduza o historico.");
            std::printf("{\"type\":\"start\",\"prompt_tokens\":%zu}\n", ids.size()); std::fflush(stdout);
            const auto t0 = std::chrono::steady_clock::now();
            en.reset();
            int pos = 0, emitted = 0, steps = 0;
            bool eos = false;
            for (int id : ids) {
                if (worker_cancelled) break;
                en.step(id, pos++);
            }
            CK(hipDeviceSynchronize());
            const auto t1 = std::chrono::steady_clock::now();
            for (int s = 0; !worker_cancelled && s < o.n_gen; ++s) {
                CK(hipMemcpy(logits.data(), en.logits, logits.size() * sizeof(float), hipMemcpyDeviceToHost));
                const int next = sample(logits, o, rng);
                if (tokenizer.is_end(next)) { eos = true; break; }
                ++emitted;
                if (!dt.is_special(next)) {
                    std::printf("{\"type\":\"token\",\"hex\":\"%s\"}\n", worker_hex(dt.piece(next)).c_str());
                    std::fflush(stdout);
                }
                if (s + 1 < o.n_gen) { en.step(next, pos++); ++steps; }
            }
            CK(hipDeviceSynchronize());
            const auto t2 = std::chrono::steady_clock::now();
            const double pre = std::chrono::duration<double>(t1 - t0).count();
            const double dec = std::chrono::duration<double>(t2 - t1).count();
            std::printf("{\"type\":\"done\",\"prompt_tokens\":%zu,\"completion_tokens\":%d,"
                        "\"prompt_per_second\":%.4f,\"predicted_per_second\":%.4f,"
                        "\"prompt_ms\":%.4f,\"decode_ms\":%.4f,\"decode_steps\":%d,"
                        "\"cancelled\":%s,\"finish_reason\":\"%s\"}\n",
                        ids.size(), emitted, pre > 0 && !worker_cancelled ? ids.size() / pre : 0,
                        dec > 0 ? steps / dec : 0, pre * 1000, dec * 1000, steps,
                        worker_cancelled ? "true" : "false", eos ? "stop" : "length");
            std::fflush(stdout);
        } catch (const std::exception& e) { worker_error(e.what()); }
    }
    return 0;
}
} // namespace lfm2
