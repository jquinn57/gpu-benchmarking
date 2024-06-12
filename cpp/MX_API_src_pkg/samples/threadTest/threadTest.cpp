#include <iostream>
#include <thread>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <fstream>
#include <chrono>
#include <ctime>
#include <condition_variable>
#include <mutex>
#include "MxAccl.h"

// comment this out to use zeros as inputs
//#define IMAGE_INPUTS

namespace fs = std::filesystem;

std::atomic_bool runflag;

// bool is_busy = false;
// std::condition_variable cond_var_ready;
// std::mutex mtx;

int total_num_frames = 1000;
int model_input_width = 0;
int model_input_height = 0;

std::vector<int64_t> in_timestamps;
std::vector<int64_t> out_timestamps;


fs::path image_path = "/home/jquinn/datasets/coco/images/val2017"; 
std::vector<fs::path> image_list;

//model info
MX::Types::MxModelInfo model_info;

//Vector to get output
std::vector<float*> ofmap;

int num_frames_processed=0;

float *ifmap0;

int input_sleep_us = 0;
int output_sleep_us = 0;


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

int64_t timestamp(){

    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return duration.count();
}

// Input callback function
bool incallback_getframe(vector<const MX::Types::FeatureMap<float>*> dst, int streamLabel){

    static int input_count = 0;

    if(input_count >= image_list.size()){
        //std::cout << "No more images to process" << std::endl;
        return false;
    }

    if(runflag.load()){


#if defined(IMAGE_INPUTS)
        // load images from ssd and resize and scale
        cv::Mat inframe = cv::imread(image_list[input_count].string());
        cv::Mat preProcframe = preprocess(inframe);
        in_timestamps.push_back(timestamp());
        dst[0]->set_data((float*)preProcframe.data, false);

#else
        // use zeros as inputs
        in_timestamps.push_back(timestamp());
        dst[0]->set_data(ifmap0, false);
#endif

        std::this_thread::sleep_for(std::chrono::microseconds(input_sleep_us));
        input_count++;
        return true;
    }
    return false;
}


// Output callback function
bool outcallback_getmxaoutput(vector<const MX::Types::FeatureMap<float>*> src, int streamLabel){

    for(int i = 0; i<model_info.num_out_featuremaps ; ++i){
        src[i]->get_data(ofmap[i], true);
    }
    out_timestamps.push_back(timestamp());
    std::this_thread::sleep_for(std::chrono::microseconds(output_sleep_us));
    num_frames_processed++;
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

float run_inference(const fs::path& dfp_path){
    runflag.store(true);
    float fps;
    in_timestamps.reserve(total_num_frames);
    out_timestamps.reserve(total_num_frames);

    if(runflag.load()){


        MX::Runtime::MxAccl accl(dfp_path.string().c_str());

        model_info = accl.get_model_info(0);
        // print_model_info();
        model_input_height = model_info.in_featuremap_shapes[0][0];
        model_input_width = model_info.in_featuremap_shapes[0][1];
        // Allocate memory to get output from accelarator
        ofmap.reserve(model_info.num_out_featuremaps);
        for(int i = 0; i<model_info.num_out_featuremaps ; ++i){
            float * fmap = new float[model_info.out_featuremap_sizes[i]];
            ofmap.push_back(fmap);
        }

        // create test input of all zeros
        int input_size = model_input_height * model_input_width * 3;
        ifmap0 = new float[input_size];
        std::memset(ifmap0, 0, input_size * sizeof(float));

        // Connect stream to acclerator
        accl.connect_stream(&incallback_getframe, &outcallback_getmxaoutput, 10 /*unique stream ID*/, 0 /*Model ID */);

        std::chrono::milliseconds start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        accl.start();
        accl.wait();
        std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;

        fps = (float)num_frames_processed * 1000 / (float)(duration.count());
        //std::cout << "duration (ms): " << duration.count() << ", num frames: " << num_frames_processed << " FPS: "<< fps << std::endl;

        accl.stop();

        // Deallocate memory used to get output
        for (auto& fmap : ofmap) {
            delete[] fmap;
            fmap = NULL;
        }
        delete[] ifmap0;
    }
    return fps; 
}


int main(int argc, char* argv[]){


    if(argc != 3){
        std::cout << "Usage: ./threadTest <input_sleep_us> <output_sleep_us> " << std::endl;
        return -1;
    }

    input_sleep_us = std::stoi(argv[1]);
    output_sleep_us = std::stoi(argv[2]);

    fs::path dfp_path = "cascadePlus/yolov5n-SiLU-416.dfp";
 
    int n = 0;
    for(const auto& entry : fs::directory_iterator(image_path)){
        if(entry.is_regular_file() && entry.path().extension() == ".jpg"){
            image_list.push_back(entry.path());
            n++;
            if(n == total_num_frames)
                break;
        }
    }

    //std::cout << "Number of images to process: " << image_list.size() << std::endl;

    signal(SIGINT, signal_handler);

    float fps = run_inference(dfp_path);
    std::cout << input_sleep_us << ", " << output_sleep_us << ", " << fps << std::endl;

    for(size_t i = 0; i < total_num_frames; i ++){
        int64_t in_dt = in_timestamps[i] - in_timestamps[0];
        int64_t out_dt = out_timestamps[i] - in_timestamps[0];
        std::cout << in_dt << ", " << out_dt << std::endl;
    }

    return 0;
}
