#include <cmath>
#include <algorithm>
#include <omp.h>

// Макросы для универсального экспорта (Windows/Linux)
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

const float MAX_AMPLITUDE = 5.0f;
// Увеличиваем базовый распад для компенсации инерции
const float BASE_DECAY = 0.001f; 

inline float softsign(float x) { return x / (1.0f + std::abs(x)); }
inline float clamp(float x) {
    if (std::isnan(x)) return 0.0f;
    return std::max(-MAX_AMPLITUDE, std::min(MAX_AMPLITUDE, x));
}

// Быстрый шум для стохастического резонанса
inline float fast_noise(int seed) {
    unsigned int x = (seed * 1103515245 + 12345);
    return ((float)(x & 0x7FFFFFFF) / 2147483648.0f) - 0.5f; 
}

// === EVOLUTION KERNEL (v37.6 - Fixed Entropy Gating) ===
EXPORT void run_avx512_evolution(
    float* __restrict__ state,      
    float* __restrict__ buffer,     
    const float* __restrict__ rules,
    const float* __restrict__ anchors, 
    int B, int D, int steps,
    float noise_level 
) {
    const float dt = 0.05f;
    const float inhib_speed = 0.4f;
    const float recovery_speed = 0.02f;

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        float* cur = &state[b * 3 * D];
        float* nxt = &buffer[b * 3 * D];
        
        for (int t = 0; t < steps; ++t) {
            float* r_in = &cur[0]; float* g_in = &cur[D]; float* b_in = &cur[2 * D];
            float* r_out = &nxt[0]; float* g_out = &nxt[D]; float* b_out = &nxt[2 * D];

            #pragma omp simd
            for (int i = 0; i < D; ++i) {
                float r = r_in[i];
                float b_fatigue = b_in[i];

                // 1. Stochastic Noise
                float noise = fast_noise(b*D*steps + i*steps + t) * noise_level;

                // 2. Entropy Gating (Homeostasis)
                // Gate теперь влияет и на память, предотвращая резонансные галлюцинации
                float gate = (b_fatigue > 0.85f) ? 0.0f : 1.0f;

                int l = (i - 1 + D) % D;
                int rt = (i + 1) % D;
                
                // Lateral Input Processing
                float input = (rules[0] * r_in[l] + rules[1] * r + rules[2] * r_in[rt]);

                // Physics (Reaction-Diffusion with Fatigue)
                float db = (std::abs(r) * inhib_speed) - (b_fatigue * recovery_speed);
                
                // Green Channel (Memory/Anchor logic)
                float dg = (anchors[i] - g_in[i]) * 0.1f + (r * 0.3f);
                
                // Red Channel (Activity)
                // Term: (input + g_in[i] * 0.6f) * gate
                // Гейтинг применяется к сумме входа и памяти
                float excitation = (input + (g_in[i] * 0.6f)) * gate;
                float inhibition = (b_fatigue * 2.5f * r);
                
                float dr = excitation - inhibition + noise;

                // Inertial clamping
                r_out[i] = clamp((r + softsign(dr) * dt) * 0.98f);
                g_out[i] = clamp(g_in[i] + dg * dt);
                b_out[i] = clamp(std::max(0.0f, b_fatigue + db * dt));
            }
            float* tmp = cur; cur = nxt; nxt = tmp;
        }
        if (steps % 2 != 0) std::copy(nxt, nxt + 3 * D, cur);
    }
}

// === MEMORY UPDATE (Zen Decay v2) ===
EXPORT void update_dynamic_memory(
    float* anchor_matrix, float* grads, int D, float lr, 
    float truth_signal, float panic_level
) {
    #pragma omp parallel for simd
    for (int d = 0; d < D; ++d) {
        float g = std::max(-1.0f, std::min(1.0f, grads[d]));
        float w = anchor_matrix[d];

        // ADS: Усиленный распад при панике
        float current_decay = BASE_DECAY * (1.0f + panic_level * 50.0f);
        
        // Hybrid L1/L2 Decay: w * (0.5 + |w|)
        // Гарантирует затухание даже малых весов (решает проблему исчезающего градиента распада)
        float decay_force = current_decay * w * (0.5f + std::abs(w));

        // Reinforcement
        float reinforcement = (truth_signal > 0.15f && g * w > 0) ? (0.05f * std::abs(g) * truth_signal) : 0.0f;

        // Update with clamp
        anchor_matrix[d] = clamp(w - ((lr * g) + decay_force - (reinforcement * ((w > 0) ? 1 : -1))));
    }
}

} // End extern "C"
// Лишняя скобка удалена
