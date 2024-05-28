#include <iostream>
#include <thread>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <fstream>
#include <chrono>
#include <ctime>
#include <thread>
#include <condition_variable>
#include <mutex>
#include "MxAccl.h"
#include "post_processor.h"


namespace fs = std::filesystem;

std::atomic_bool runflag;

fs::path model_path = "cascadePlus/yolov5n-SiLU-416.dfp";
fs::path onnx_model_path = "model_0_yolov5n-SiLU-416_post.onnx";

int total_num_frames = 1000;
int model_input_width = 416;
int model_input_height = 416;

fs::path image_path = "/home/jquinn/datasets/coco/images/val2017"; 
const char* const output_node_names[] = {"output0"};

std::vector<fs::path> image_list;


//model info
MX::Types::MxModelInfo model_info;


//Queue to add input frames
std::deque<cv::Mat> frames_queue;
std::mutex frame_queue_lock;
//Queue to add output from mxa
std::deque<std::vector<float*>> ofmap_queue;
std::mutex ofmap_queue_lock;
std::condition_variable cond_var;
std::condition_variable cond_var_out_empty;
std::condition_variable cond_var_out_not_empty;


//Vector to get output
std::vector<float*> ofmap;

int num_frames_processed=0;


Ort::MemoryInfo PostProcessor::s_memoryInfo =  Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

PostProcessor::PostProcessor(const std::filesystem::path &model_path)
{
    std::cout << "Loading model: " << model_path.string() << std::endl;
    // Onnx Threading option to limit CPU usage
    OrtEnv* environment;
    OrtThreadingOptions* envOpts = nullptr;
    Ort::GetApi().CreateThreadingOptions(&envOpts);
    Ort::GetApi().SetGlobalIntraOpNumThreads(envOpts, 2);
    Ort::GetApi().SetGlobalInterOpNumThreads(envOpts, 2);
    Ort::GetApi().SetGlobalSpinControl(envOpts, 0);
    Ort::GetApi().CreateEnvWithGlobalThreadPools(ORT_LOGGING_LEVEL_WARNING, "objectDetectionSample", envOpts, &environment);
    
    m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "post-processing");
    m_session_options.SetIntraOpNumThreads(2);
    m_session_options.SetInterOpNumThreads(2);
    m_session_ptr = new Ort::Session(m_env, model_path.c_str(), m_session_options);

}

PostProcessor::~PostProcessor()
{
    std::cout << "Post processor shut down" << std::endl;
    delete m_session_ptr;
}


void PostProcessor::process(const std::vector<float *> &output)
{
    static int completed_frames = 0;

    std::vector<Ort::Value> inputs;
    for(int i=0; i<model_info.num_out_featuremaps; ++i){
        auto shape = model_info.out_featuremap_shapes[i].chfirst_shape();
        inputs.push_back(Ort::Value::CreateTensor<float>(s_memoryInfo,
                                                        output[i],
                                                        model_info.out_featuremap_sizes[i],
                                                        shape.data(),
                                                        shape.size()));
    }

    auto outputTensors =  m_session_ptr->Run(Ort::RunOptions{}, 
                                        model_info.output_layer_names.data(), 
                                        inputs.data(), 
                                        model_info.num_out_featuremaps, 
                                        output_node_names, 1);

    completed_frames++;
    // std::cout << "post processing: " << completed_frames << std::endl;

}

//signal handler
void signal_handler(int p_signal){
    runflag.store(false);
}

// frame preprocessing
cv::Mat preprocess( const cv::Mat& image ) {

    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(model_input_height, model_input_width), 0, 0, cv::INTER_LINEAR);

    // Convert image to float32 and normalize
    cv::Mat floatImage;
    resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    return floatImage;
}


// Input callback function
bool incallback_getframe_good(vector<const MX::Types::FeatureMap<float>*> dst, int streamLabel){

    static int count = 0;

    if(count >= image_list.size()){
        std::cout << "No more files to load" << std::endl;
        return false;
    }

    if(runflag.load()){

        cv::Mat inframe = cv::imread(image_list[count].string());
        cv::Mat preProcframe = preprocess(inframe);

        // Set preprocessed input data to be sent to accelarator
        dst[0]->set_data((float*)preProcframe.data, false);
        count++;
        return true;
    }
    return false;
}


void post_processing_worker(){


    std::unique_ptr<PostProcessor> pp = std::make_unique<PostProcessor>(onnx_model_path);;

    while(runflag.load()){

        std::unique_lock<std::mutex> olock(ofmap_queue_lock);
        cond_var_out_not_empty.wait(olock, [] { return !ofmap_queue.empty(); });
        auto output = ofmap_queue.front();
        pp->process(output);
        num_frames_processed++;
        ofmap_queue.pop_front();
        cond_var_out_empty.notify_one();

        if(num_frames_processed >= total_num_frames)
            runflag.store(false);

    }

}


// Output callback function
bool outcallback_getmxaoutput(vector<const MX::Types::FeatureMap<float>*> src, int streamLabel){

    {
        std::unique_lock<std::mutex> olock(ofmap_queue_lock);
        // wait until the ofmap_queue is empty 
        // because there is only enough memory allocated to hold one set of outputs
        // we could just send the output to post_processor in this same thread
        // but for some reason even creating the onnx session caused FPS to drop to 35 (without actually doing the post-processing)

        cond_var_out_empty.wait(olock, [] { return ofmap_queue.empty(); });

        for(int i = 0; i<model_info.num_out_featuremaps ; ++i){
            src[i]->get_data(ofmap[i], true);
        }
        ofmap_queue.push_back(ofmap);
        cond_var_out_not_empty.notify_one();
    }
    return true;
}



void print_model_info(){
    std::cout << "\n******** Model Index : " << model_info.model_index << " ********\n";
    std::cout << "\nNum of in featureMaps : " << model_info.num_in_featuremaps << "\n";
    
    std::cout << "\nIn featureMap Shapes \n";
    for(int i = 0; i<model_info.num_in_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        std::cout << "Layer Name : " << model_info.input_layer_names[i] << "\n";
        std::cout << "H = " << model_info.in_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << model_info.in_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << model_info.in_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << model_info.in_featuremap_shapes[i][3] << "\n";
    }

    std::cout << "\n\nNum of out featureMaps : " << model_info.num_out_featuremaps << "\n";
    std::cout << "\nOut featureMap Shapes \n";
    for(int i = 0; i<model_info.num_out_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        // std::cout << "Layer Name : " << model_info.output_layer_names[i] << "\n";
        std::cout << "H = " << model_info.out_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << model_info.out_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << model_info.out_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << model_info.out_featuremap_shapes[i][3] << "\n";
    }

    std::cout << "Done printing model info \n";

}

void run_inference(){
    runflag.store(true);

    if(runflag.load()){

        std::thread post_processing_thread(post_processing_worker);

        MX::Runtime::MxAccl accl(model_path.c_str());

        model_info = accl.get_model_info(0);
        print_model_info();
        model_input_height = model_info.in_featuremap_shapes[0][0];
        model_input_width = model_info.in_featuremap_shapes[0][1];
        // Allocate memory to get output from accelarator
        ofmap.reserve(model_info.num_out_featuremaps);
        for(int i = 0; i<model_info.num_out_featuremaps ; ++i){
            float * fmap = new float[model_info.out_featuremap_sizes[i]];
            ofmap.push_back(fmap);
        }

        // Connect stream to acclerator
        accl.connect_stream(&incallback_getframe_good, &outcallback_getmxaoutput, 10 /*unique stream ID*/, 0 /*Model ID */);
        std::cout << "Connected stream \n\n\n";

        std::chrono::milliseconds start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        
        accl.start();

        accl.wait();
        post_processing_thread.join();
        std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;

        float fps = (float)num_frames_processed * 1000 / (float)(duration.count());
        std::cout << "duration (ms): " << duration.count() << ", num frames: " << num_frames_processed << " FPS: "<< fps << std::endl;

        accl.stop();
        std::cout << "\nAccl stop called \n";

        // Deallocate memory used to get output
        for (auto& fmap : ofmap) {
            delete[] fmap;
            fmap = NULL;
        }
    }    
}


int main(int argc, char* argv[]){

    int n = 0;
    for(const auto& entry : fs::directory_iterator(image_path)){
        if(entry.is_regular_file() && entry.path().extension() == ".jpg"){
            image_list.push_back(entry.path());
            n++;
            if(n == total_num_frames)
                break;
        }
    }
    std::cout << "Number of images to process: " << image_list.size() << std::endl;

    signal(SIGINT, signal_handler);
    runflag.store(true);

    if(runflag.load()){

        std::cout << "application start \n";
        std::cout << "model path = " << model_path.c_str() << "\n";

        run_inference();

    }

    else{
        std::cout << "App exiting without execution \n\n\n";       
    }
    
    return 1;
}
