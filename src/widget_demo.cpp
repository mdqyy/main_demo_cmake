#include "widget_demo.h"
#include "ui_widget_demo.h"

#include <cstdio>
#define LOG(A) //
using namespace std;
using namespace cv;
namespace
{
bool c_list_gray_flag = false;
}



#include <boost/foreach.hpp>


WidgetDemo::WidgetDemo(int argc, char *argv[], QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WidgetDemo),
    m_ts_detect("cascade.xml"),
    m_ex_obj_detector( new doppia::ObjectsDetectionApplication() )
{
    m_ex_obj_detector->prepare(argc,argv);
    ui->setupUi(this);
    initProcess();
    LOG(cout<<"after init class"<<endl;)
}

void WidgetDemo::initProcess()
{
    m_video.open("/home/yerrick/shared_Development/TrafficSign/material/fc2_save_2013-03-22-134254-0000.avi");//e6.mp4");
    if(!m_video.isOpened())
    {
        cout<<"Error in video input !!"<<std::endl;
    }
    Mat init;
    m_video>>init;
    drawDetectResults(init);


}

WidgetDemo::~WidgetDemo()
{
    delete ui;
}



void WidgetDemo::keyPressEvent(QKeyEvent *key)
{
    LOG(cout<<"in key event"<<endl;)
            switch(key->key())
    {
        case Qt::Key_B:
            updateFrame();
            break;
        default:
            break;
    }
}



void WidgetDemo::mousePressEvent(QKeyEvent *mouse)
{

    LOG(cout<<"in mouse press event"<<endl;)

}




void WidgetDemo::updateFrame()
{
    LOG(cout<<"start update frame"<<endl);

    /// new way
    m_ex_obj_detector->one_step();
    Mat temp = m_ex_obj_detector->get_image();
    const doppia::AbstractObjectsDetector::detections_t &detect_result = m_ex_obj_detector->get_detections();


    /// opencv ways
    //    vector<Rect> detect_result;
    //    m_video >> temp;
    //    LOG(cout<<"start detect"<<endl);
    //    m_ts_detect.Detect(temp,detect_result);
    //    LOG(cout<<"start draw"<<endl);

    drawDetectResults(temp,detect_result);
    listRecogniseResults(temp,detect_result);

}
void WidgetDemo::drawDetectResults(const Mat &input_image)
{
    LOG(cout<<"is drawing"<<endl);
    // convert image to 3 channel
    switch(input_image.channels())
    {
    case 1:
        cvtColor(input_image, input_image, CV_GRAY2BGR);
        break;
    default:
        break;
    }

    LOG(cout<<"before draw original image"<<endl);
    // show original image
    QImage img = QImage(input_image.data,input_image.cols,input_image.rows,QImage::Format_RGB888);
    img = img.rgbSwapped();
    QImage show1;
    show1 = img;
    show1 = show1.scaled(ui->labelOriginal->width(),ui->labelOriginal->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    QPixmap temp;
    temp.convertFromImage(show1);
    ui->labelOriginal->setPixmap(temp);

    LOG(cout<<"before draw processed image"<<endl);
    // show original image + traffic signs detected
    QImage show2 = img;
    show2 = show2.scaled(ui->labelAfterProcess->width(),ui->labelAfterProcess->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    temp.convertFromImage(show2);
    ui->labelAfterProcess->setPixmap(temp);

    this->update();
}

void WidgetDemo::drawDetectResults(const Mat &input_image, const std::vector< cv::Rect> & detections)
{
    // convert image to 3 channel
    LOG(cout<<"is drawing"<<endl);
    switch(input_image.channels())
    {
    case 1:
        cvtColor(input_image, input_image, CV_GRAY2BGR);
        break;
    default:
        break;
    }
    LOG(cout<<"before draw original image"<<endl);
    // show original image
    QImage img = QImage(input_image.data,input_image.cols,input_image.rows,QImage::Format_RGB888);
    //    img = img.rgbSwapped();
    QImage show1;
    show1 = img;
    show1 = show1.scaled(ui->labelOriginal->width(),ui->labelOriginal->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    QPixmap temp1;
    temp1.convertFromImage(show1);
    ui->labelOriginal->setPixmap(temp1);

    LOG(cout<<"before draw processed image"<<endl);
    // show original image + traffic signs detected
    QImage show2 = img;
    QPixmap temp2;
    QPainter tpainter(&show2);
    QPen tpen;
    tpen.setWidth(5);
    tpen.setColor(Qt::blue);
    tpainter.setPen(tpen);
    Rect trect;
    for(size_t i=0;i<detections.size();i++)
    {
        trect = detections[i];
        tpainter.drawRect(trect.x,trect.y,trect.width,trect.height);
    }
    LOG(cout<<"before scale image"<<endl);
    QImage show3 = show2.scaled(ui->labelAfterProcess->width(),ui->labelAfterProcess->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    temp2.convertFromImage(show3);
    ui->labelAfterProcess->setPixmap(temp2);

    this->update();
}

void WidgetDemo::drawDetectResults(const cv::Mat &input_image, const doppia::AbstractObjectsDetector::detections_t &detections)
{
    typedef doppia::AbstractObjectsDetector::detection_t detection_type;
    float min_score = 0; // we will saturate at negative scores
    float max_detection_score = 0;
    float additional_border = 0;
    //float min_score = std::numeric_limits<float>::max();
    // show original image + traffic signs detected

    // convert image to 3 channel
    LOG(cout<<"is drawing"<<endl);
    switch(input_image.channels())
    {
    case 1:
        cvtColor(input_image, input_image, CV_GRAY2BGR);
        break;
    default:
        break;
    }
    LOG(cout<<"before draw original image"<<endl);


    // show original image
    QImage img = QImage(input_image.data,input_image.cols,input_image.rows,QImage::Format_RGB888);
    //    img = img.rgbSwapped();
    QImage show1;
    show1 = img;
    show1 = show1.scaled(ui->labelOriginal->width(),ui->labelOriginal->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    QPixmap temp1;
    temp1.convertFromImage(show1);
    ui->labelOriginal->setPixmap(temp1);
    LOG(cout<<"before draw processed image"<<endl);



    // show original + traffic signs detected
    QImage show2 = img;
    QPixmap temp2;
    QPainter tpainter(&show2);
    QPen tpen;
    tpen.setWidth(5);
    tpen.setColor(Qt::blue);
    tpainter.setPen(tpen);
    BOOST_FOREACH(const detection_type &detection, detections)
    {
        max_detection_score = std::max(max_detection_score, detection.score);
    }
    const float scaling = 255 / (max_detection_score - min_score);
    BOOST_FOREACH(const detection_type &detection, detections)
    {
        const boost::uint8_t normalized_score = static_cast<boost::uint8_t>(
                    std::max(0.0f, (detection.score - min_score)*scaling));
        //boost::gil::rgb8c_pixel_t color = rgb8_colors::white;
        boost::gil::rgb8c_pixel_t color(normalized_score, 0, 0);

        detection_type::rectangle_t box = detection.bounding_box;

        box.min_corner().x(box.min_corner().x() - additional_border);
        box.min_corner().y(box.min_corner().y() - additional_border);
        box.max_corner().x(box.max_corner().x() - additional_border);
        box.max_corner().y(box.max_corner().y() - additional_border);



        tpainter.drawRect(box.min_corner().x(),
                          box.min_corner().y(),
                          box.max_corner().x() - box.min_corner().x(),
                          box.max_corner().y() - box.min_corner().y());
    }





    LOG(cout<<"before scale image"<<endl);
    QImage show3 = show2.scaled(ui->labelAfterProcess->width(),ui->labelAfterProcess->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    temp2.convertFromImage(show3);
    ui->labelAfterProcess->setPixmap(temp2);






    // draw ground truth
    //    for(size_t i=0; i < ground_truth_detections.size(); i+=1)
    //    {
    //        boost::gil::rgb8c_pixel_t color = rgb8_colors::white;
    //        const detection_t::rectangle_t &box = ground_truth_detections[i].bounding_box;
    //        draw_rectangle(view, color, box, 2);
    //    }

    return;
}

void WidgetDemo::listRecogniseResults(const Mat &input_image, const std::vector<Rect> &rect)
{
    ui->listFromImage->clear();

    LOG(cout<<"is listing"<<endl;)
            Mat timage;
    switch(input_image.channels())
    {
    case 1:
        cvtColor(input_image, input_image, CV_GRAY2BGR);
        break;
    default:
        if(c_list_gray_flag)
        {
            cvtColor(input_image, timage, CV_BGR2GRAY);
            cvtColor(timage, timage, CV_GRAY2BGR);
        }
        break;
    }
    QImage img = QImage(timage.data,timage.cols,timage.rows,QImage::Format_RGB888);
    img = img.rgbSwapped();

    LOG(cout<<"before cut image"<<endl;)

            for(size_t i=0;i<rect.size();i++)
    {
        QImage roi = img.copy(rect[i].x,rect[i].y,rect[i].width,rect[i].height);
        QPixmap pix;
        pix.convertFromImage(roi);
        QIcon icon(pix);
        QListWidgetItem *item = new QListWidgetItem(icon,QString::number(i));
        ui->listFromImage->addItem(item);
    }
    this->update();
}


void WidgetDemo::listRecogniseResults(const Mat &input_image, const doppia::AbstractObjectsDetector::detections_t &detections)
{
    ui->listFromImage->clear();

    LOG(cout<<"is listing"<<endl;)
            Mat timage;
    switch(input_image.channels())
    {
    case 1:
        cvtColor(input_image, input_image, CV_GRAY2BGR);
        break;
    default:
        if(c_list_gray_flag)
        {
            cvtColor(input_image, timage, CV_BGR2GRAY);
            cvtColor(timage, input_image, CV_GRAY2BGR);
        }
        break;
    }
    QImage img = QImage(input_image.data,input_image.cols,input_image.rows,QImage::Format_RGB888);


    LOG(cout<<"before cut image"<<endl);

    typedef doppia::AbstractObjectsDetector::detection_t detection_type;
    int i=-1;
    BOOST_FOREACH(const detection_type &detection, detections)
    {
        i++;
        detection_type::rectangle_t box = detection.bounding_box;
        QImage roi = img.copy(box.min_corner().x(),
                              box.min_corner().y(),
                              box.max_corner().x() - box.min_corner().x(),
                              box.max_corner().y() - box.min_corner().y());
        QPixmap pix;
        pix.convertFromImage(roi);
        QIcon icon(pix);
        QListWidgetItem *item = new QListWidgetItem(icon,QString::number(i));
        ui->listFromImage->addItem(item);
    }
    this->update();
}















