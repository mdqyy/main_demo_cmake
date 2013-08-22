#include "widget_demo.h"
#include <QApplication>



int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    WidgetDemo w(argc, argv);
    w.show();
    return a.exec();
}
