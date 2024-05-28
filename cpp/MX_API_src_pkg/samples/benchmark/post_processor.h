#pragma once


#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <provider_options.h>

class PostProcessor
{
    public:
        PostProcessor(const std::filesystem::path &model_path);
        ~PostProcessor();
        void process(const std::vector<float *> &output);
    private:
        Ort::Env m_env;
        Ort::SessionOptions m_session_options;
        Ort::Session *m_session_ptr;
        static Ort::MemoryInfo s_memoryInfo;

};


