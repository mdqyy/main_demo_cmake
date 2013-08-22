#ifndef WIDGET_DEMO_H
#define WIDGET_DEMO_H


#include <QtGui>
#include <opencv2/opencv.hpp>
#include <vector>
#include "object_detector/traffic_sign_detector.h"


#include "objects_detection/AbstractObjectsDetector.hpp"
#include "objects_detection_app/ObjectsDetectionApplication.hpp"
#include <boost/scoped_ptr.hpp>



namespace Ui {
class WidgetDemo;
}

class WidgetDemo : public QWidget
{
    Q_OBJECT
    
public:
    explicit WidgetDemo(int argc,char* argv[],QWidget *parent = 0);
    ~WidgetDemo();

    void keyPressEvent(QKeyEvent *key);
    void mousePressEvent(QKeyEvent *mouse);

public:
    void initProcess();
    void updateFrame();
    void drawDetectResults(const cv::Mat &input_image);
    void drawDetectResults(const cv::Mat &input_image, const std::vector< cv::Rect> & temp);
    void drawDetectResults(const cv::Mat &input_image, const doppia::AbstractObjectsDetector::detections_t &detections);
    void listRecogniseResults(const cv::Mat &input_image, const std::vector< cv::Rect> &input_rect);
    void listRecogniseResults(const cv::Mat &input_image, const doppia::AbstractObjectsDetector::detections_t &input_rect);

private:
    Ui::WidgetDemo *ui;
    cv::VideoCapture m_video;
    TrafficSignDetector m_ts_detect;
    int m_cycle;


    boost::scoped_ptr<doppia::ObjectsDetectionApplication>  m_ex_obj_detector;
};



#endif // WIDGET_DEMO_H
