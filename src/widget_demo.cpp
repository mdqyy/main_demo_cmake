#include "widget_demo.h"
#include "ui_widget_demo.h"

#include <cstdio>
#define LOG(A) //
using namespace std;
using namespace cv;
namespace
{
bool c_list_gray_flag = true;
}
WidgetDemo::WidgetDemo(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WidgetDemo),
    m_ts_detect("cascade.xml")
{
    ui->setupUi(this);
    initProcess();
    LOG(cout<<"after init class"<<endl;)
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


void WidgetDemo::initProcess()
{
    m_video.open("e6.mp4");
    if(!m_video.isOpened())
    {
        cout<<"Error in video input !!"<<std::endl;
    }
    Mat init;
    m_video>>init;
    drawDetectResults(init);
}

void WidgetDemo::updateFrame()
{
    LOG(cout<<"start update frame"<<endl;)
            Mat temp;
    vector<Rect> detect_result;
    m_video >> temp;
    LOG(cout<<"start detect"<<endl;)
            m_ts_detect.Detect(temp,detect_result);
    LOG(cout<<"start draw"<<endl;)
            drawDetectResults(temp,detect_result);
    listRecogniseResults(temp,detect_result);

}
void WidgetDemo::drawDetectResults(const Mat &input_image)
{
    LOG(cout<<"is drawing"<<endl;)
            // convert image to 3 channel
            switch(input_image.channels())
    {
        case 1:
            cvtColor(input_image, input_image, CV_GRAY2BGR);
            break;
        default:
            break;
    }

    LOG(cout<<"before draw original image"<<endl;)
            // show original image
            QImage img = QImage(input_image.data,input_image.cols,input_image.rows,QImage::Format_RGB888);
    img = img.rgbSwapped();
    QImage show1;
    show1 = img;
    show1 = show1.scaled(ui->labelOriginal->width(),ui->labelOriginal->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    QPixmap temp;
    temp.convertFromImage(show1);
    ui->labelOriginal->setPixmap(temp);

    LOG(cout<<"before draw processed image"<<endl;)
            // show original image + traffic signs detected
            QImage show2 = img;
    show2 = show2.scaled(ui->labelAfterProcess->width(),ui->labelAfterProcess->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    temp.convertFromImage(show2);
    ui->labelAfterProcess->setPixmap(temp);

    this->update();
}

void WidgetDemo::drawDetectResults(const Mat &input_image, const std::vector<Rect>& result)
{
    // convert image to 3 channel
    LOG(cout<<"is drawing"<<endl;)
            switch(input_image.channels())
    {
        case 1:
            cvtColor(input_image, input_image, CV_GRAY2BGR);
            break;
        default:
            break;
    }
    LOG(cout<<"before draw original image"<<endl;)
            // show original image
            QImage img = QImage(input_image.data,input_image.cols,input_image.rows,QImage::Format_RGB888);
    img = img.rgbSwapped();
    QImage show1;
    show1 = img;
    show1 = show1.scaled(ui->labelOriginal->width(),ui->labelOriginal->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    QPixmap temp1;
    temp1.convertFromImage(show1);
    ui->labelOriginal->setPixmap(temp1);

    LOG(cout<<"before draw processed image"<<endl;)
            // show original image + traffic signs detected
            QImage show2 = img;
    QPixmap temp2;
    QPainter tpainter(&show2);
    QPen tpen;
    tpen.setWidth(5);
    tpen.setColor(Qt::blue);
    tpainter.setPen(tpen);
    Rect trect;
    for(size_t i=0;i<result.size();i++)
    {
        trect = result[i];
        tpainter.drawRect(trect.x,trect.y,trect.width,trect.height);
    }
    LOG(cout<<"before scale image"<<endl;)
            QImage show3 = show2.scaled(ui->labelAfterProcess->width(),ui->labelAfterProcess->height(),Qt::KeepAspectRatio,Qt::FastTransformation);
    temp2.convertFromImage(show3);
    ui->labelAfterProcess->setPixmap(temp2);

    this->update();
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















