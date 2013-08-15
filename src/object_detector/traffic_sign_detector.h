#ifndef TRAFFIC_SIGN_DETECTOR_H
#define TRAFFIC_SIGN_DETECTOR_H

#include <opencv2/opencv.hpp>

// basic class as interface
class ObjectDetector
{
public:
    ObjectDetector(){};
    ~ObjectDetector(){};
    virtual void Detect(cv::Mat input_image,std::vector< cv::Rect >& output_rect)=0;
};

class ObjectRecognizer
{
public:
    ObjectRecognizer(){};
    ~ObjectRecognizer(){};
    virtual void Recognize(cv::Mat input_image,int output_label)=0;
};


// for traffic signs
class TrafficSignDetector : public ObjectDetector
{
public:
    TrafficSignDetector(std::string name);
    ~TrafficSignDetector(){};
    virtual void Detect(cv::Mat input_image, std::vector<cv::Rect>& output_rect);

private:
    cv::CascadeClassifier m_cascade_dect;
};
class TrafficSignRecognizer : public ObjectRecognizer
{
public:
    TrafficSignRecognizer(){};
    ~TrafficSignRecognizer(){};
    void Recognize(cv::Mat input_image, int output_label);
};

#endif // TRAFFIC_SIGN_DETECTOR_H
