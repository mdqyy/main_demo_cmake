#include "traffic_sign_detector.h"
#include <cstdio>
using namespace cv;
TrafficSignDetector::TrafficSignDetector(string name)
{
    if( !m_cascade_dect.load(name) )
    {
        printf("Error loading in traffic sign detection \n");
    }
}
void TrafficSignDetector::Detect(Mat input_image, std::vector<Rect> &output_rect)
{
    Mat gray;
    cvtColor( input_image, gray, CV_BGR2GRAY );
    equalizeHist( gray, gray );
    m_cascade_dect.detectMultiScale( gray, output_rect, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(5, 5),Size(100,100) );
}
