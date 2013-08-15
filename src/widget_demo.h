#ifndef WIDGET_DEMO_H
#define WIDGET_DEMO_H


#include <QtGui>
#include <opencv2/opencv.hpp>
#include <vector>
#include "object_detector/traffic_sign_detector.h"






namespace Ui {
class WidgetDemo;
}

class WidgetDemo : public QWidget
{
    Q_OBJECT
    
public:
    explicit WidgetDemo(QWidget *parent = 0);
    ~WidgetDemo();

    void keyPressEvent(QKeyEvent *key);
    void mousePressEvent(QKeyEvent *mouse);

public:
    void initProcess();
    void updateFrame();
    void drawDetectResults(const cv::Mat &input_image);
    void drawDetectResults(const cv::Mat &input_image, const std::vector< cv::Rect> & temp);
    void listRecogniseResults(const cv::Mat &input_image, const std::vector< cv::Rect> &input_rect);

private:
    Ui::WidgetDemo *ui;
    cv::VideoCapture m_video;
    TrafficSignDetector m_ts_detect;
    int m_cycle;
};



#endif // WIDGET_DEMO_H
