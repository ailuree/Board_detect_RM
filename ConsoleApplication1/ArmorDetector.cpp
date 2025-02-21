#include "ArmorDetector.h"
#include <cmath>
#include <iostream>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core.hpp> 
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

int main()
{
    Mat src; // 原图
    Mat dst_BR; // 调节亮度后image
    Mat dst; // 二值化后的image
    vector<RotatedRect> vContour; // 发现的旋转矩形
    vector<RotatedRect> vRec; // 发现的装甲板

    //char Image_RED[] = "video\\1.MOV";
    //char Image_BLUE[] = "video\\2.MOV";
    
    VideoCapture cap("E:\\vgd\\shibie\\test.mp4");//用opencv打开本地视频文件
    //VideoWriter writer("..\\BLUE.avi", CAP_OPENCV_MJPEG, 25.0, Size(1920,1080));
    //src = imread("C:\\Users\\69022\\Desktop\\Rm\\text.png");
    namedWindow("original cap", CV_WINDOW_NORMAL);//窗口1 原始视频
    namedWindow("end cap1", CV_WINDOW_NORMAL);//窗口2 灰度化
    namedWindow("end cap2", CV_WINDOW_NORMAL);//窗口3 识别标定
    while (1) {
        cap >> src;//读取每1帧
        if (src.empty()) {
            break;// 如果为空帧,跳出循环
        }
        imshow("original cap", src); // 显示原始帧
        //resize(src, src, Size(1470, 810));
        dst_BR = Adjust_Bright(src);// 调节亮度
        dst = Threshold_Demo(dst_BR);// 图像二值化
        bool vContour_BP = Found_Contour(dst, vContour); // 寻找灯柱
        bool vRec_BP = 0;
        if (vContour_BP) {
            vRec_BP = Identify_board(dst, vContour, vRec);// 识别装甲板
        }
        if (vRec_BP)
            drawBox(vRec, src, dst);// 绘制识别结果
        drawBox1(vContour, src, dst); // 绘制灯柱结果
        imshow("end cap1", dst);// 显示二值化图像
        //writer << src;
        imshow("end cap2", src);// 显示绘制结果的原图
        waitKey(10);
    }
    waitKey(0);
    return 0;
}

Mat Adjust_Bright(Mat src)//调整亮度
{
    Mat dst_BR = Mat::zeros(src.size(), src.type()); // 创建一个与src大小和类型相同的全0矩阵dst_BR
    Mat BrightnessLut(1, 256, CV_8UC1);// 创建一个1行256列的8位无符号整数查找表
    for (int i = 0; i < 256; i++) {
        BrightnessLut.at<uchar>(i) = saturate_cast<uchar>(i + _Armor.Image_bright);
    }// 使用saturate_cast将调整后的像素值限制在[0,255]范围内,存入查找表
    LUT(src, BrightnessLut, dst_BR);// 使用查找表改变src的亮度,结果存入dst_BR
    return dst_BR; // 返回亮度调整后的图像
}

Mat Threshold_Demo(Mat dst_BR)//膨胀二值化
{
    Mat dst, channels[3];
    // 分离通道
    split(dst_BR, channels);
    if (_Armor.Armor_Color)
        dst = (channels[2] - channels[1]) * 2; // 如果是蓝色装甲板,取R通道减B通道差值的两倍为目标图像
    else
        dst = channels[0] - channels[2];  // 如果是红色装甲板,取B通道减R通道差值为目标图像
    blur(dst, dst, Size(1, 3)); // 沿垂直方向模糊目标图像
    threshold(dst, dst, _Armor.threshold_value, 255, THRESH_BINARY); // 二值化目标图像

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3), Point(-1, -1)); // 创建椭圆结构元素
    dilate(dst, dst, kernel); // 膨胀
    return dst;
}

RotatedRect& adjustRec(RotatedRect& rec)//将矩形的角度调整到-45到45度之间
{
    while (rec.angle >= 90.0)
        rec.angle -= 180.0;// 将角度调整到-90到90度之间
    while (rec.angle < -90.0)
        rec.angle += 180.0;
    if (rec.angle >= 45.0) {
        swap(rec.size.width, rec.size.height); // 交换宽高
        rec.angle -= 90.0;// 旋转角度减90度
    }
    else if (rec.angle < -45.0) {
        swap(rec.size.width, rec.size.height);
        rec.angle += 90.0;// 旋转角度加90度
    }
    return rec; // 返回调整后的矩形
}

bool Found_Contour(Mat dst, vector<RotatedRect>& vContour)
{
    vContour.clear(); // 清空储存轮廓的向量
    vector<vector<Point>> Light_Contour; // 定义轮廓点向量
    findContours(dst, Light_Contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // 寻找轮廓,只取外轮廓,只保留拐点


    for (int i = 0; i < Light_Contour.size(); i++) {// 求轮廓面积
        float Light_Contour_Area = contourArea(Light_Contour[i]);// 去除较小轮廓&fitEllipse的限制条件
        if (Light_Contour_Area < _Armor.Light_Area_min || Light_Contour[i].size() <= 5)
            continue;// 用椭圆拟合区域得到外接矩形
        RotatedRect Light_Rec = fitEllipse(Light_Contour[i]);
        Light_Rec = adjustRec(Light_Rec);
        if (Light_Rec.angle > _Armor.Light_angle)
            continue;// 长宽比和凸度限制
        if (Light_Rec.size.width / Light_Rec.size.height > _Armor.Light_Aspect_ratio
            || Light_Contour_Area / Light_Rec.size.area() < _Armor.Light_crown)
            continue;// 扩大灯柱的面积
        Light_Rec.size.height *= 1.1;
        Light_Rec.size.width *= 1.1;
        vContour.push_back(Light_Rec);// 将符合条件的轮廓存入向量
    }
    if (vContour.empty()) {
        cout << "not found Contour!!" << endl;
        return false;
    }
    if (vContour.size() < 2) {
        cout << "Contour is less!!" << endl;
        return false;
    }
    return true;
}

bool Identify_board(Mat dst, vector<RotatedRect>& vContour, vector<RotatedRect>& vRec)
{
    vRec.clear(); //清空装甲板向量
    for (int i = 0; i < vContour.size(); i++) {
        for (int j = i + 1; j < vContour.size(); j++) {

            //判断是否为相同灯条
            float Contour_angle = abs(vContour[i].angle - vContour[j].angle); //计算角度差
            if (Contour_angle >= _Armor.Light_Contour_angle)
                continue; // 角度差大于阈值, 跳过
            //长度差比率
            float Contour_Len1 = abs(vContour[i].size.height - vContour[j].size.height) / max(vContour[i].size.height, vContour[j].size.height);
            //宽度差比率
            float Contour_Len2 = abs(vContour[i].size.width - vContour[j].size.width) / max(vContour[i].size.width, vContour[j].size.width);
            if (Contour_Len1 > _Armor.Light_Contour_Len || Contour_Len2 > _Armor.Light_Contour_Len)
                continue;
            //装甲板匹配
            RotatedRect Rect;
            Rect.center.x = (vContour[i].center.x + vContour[j].center.x) / 2.; //x坐标
            Rect.center.y = (vContour[i].center.y + vContour[j].center.y) / 2.; //y坐标
            Rect.angle = (vContour[i].angle + vContour[j].angle) / 2.; //角度
            float nh, nw, yDiff, xDiff;
            nh = (vContour[i].size.height + vContour[j].size.height) / 2; //高度
            // 宽度
            nw = sqrt((vContour[i].center.x - vContour[j].center.x) * (vContour[i].center.x - vContour[j].center.x) + (vContour[i].center.y - vContour[j].center.y) * (vContour[i].center.y - vContour[j].center.y));
            float ratio = nw / nh; // 匹配到的装甲板的长宽比
            xDiff = abs(vContour[i].center.x - vContour[j].center.x) / nh; //x差比率
            yDiff = abs(vContour[i].center.y - vContour[j].center.y) / nh; //y差比率
            Rect = adjustRec(Rect);//调整矩形角度
            if (Rect.angle > _Armor.Armor_angle_min || ratio < _Armor.Armor_ratio_min || ratio > _Armor.Armor_ratio_max || xDiff < _Armor.Armor_xDiff || yDiff > _Armor.Armor_yDiff)
                continue;
            Rect.size.height = nh;
            Rect.size.width = nw;
            vRec.push_back(Rect); //保存符合的装甲板矩形
        }
    }
    if (vRec.empty()) {
        cout << "not found Rec!!" << endl;
        return false; //返回false;
    }
    return true;
}

void drawBox1(vector<RotatedRect>& vRec, Mat& src, Mat& drc)
{
    Point2f pt[4];
    for (int i = 0; i < vRec.size(); i++) {
        RotatedRect Rect = vRec[i]; // 获取一个装甲板
        Rect.points(pt); // 获取装甲板的四个顶点
        // 在两个图像上画出装甲板的轮廓
        line(src, pt[0], pt[1], CV_RGB(255, 0, 0), 1, 8, 0);
        line(src, pt[1], pt[2], CV_RGB(255, 0, 0), 1, 8, 0);
        line(src, pt[2], pt[3], CV_RGB(255, 0, 0), 1, 8, 0);
        line(src, pt[3], pt[0], CV_RGB(255, 0, 0), 1, 8, 0);
        line(drc, pt[0], pt[1], CV_RGB(255, 0, 0), 1, 8, 0);
        line(drc, pt[1], pt[2], CV_RGB(255, 0, 0), 1, 8, 0);
        line(drc, pt[2], pt[3], CV_RGB(255, 0, 0), 1, 8, 0);
        line(drc, pt[3], pt[0], CV_RGB(255, 0, 0), 1, 8, 0);
    }
}

void drawBox(vector<RotatedRect>& vRec, Mat& src, Mat& drc)
{
    Point2f pt[4];
    for (int i = 0; i < vRec.size(); i++) {
        RotatedRect Rect = vRec[i]; // 获取一个装甲板
        Rect.points(pt);// 获取装甲板的四个顶点
        // 在src上画出装甲板的轮廓,粉色,线宽2
        line(src, pt[0], pt[1], CV_RGB(255, 0, 255), 2, 8, 0);
        line(src, pt[1], pt[2], CV_RGB(255, 0, 255), 2, 8, 0);
        line(src, pt[2], pt[3], CV_RGB(255, 0, 255), 2, 8, 0);
        line(src, pt[3], pt[0], CV_RGB(255, 0, 255), 2, 8, 0);
        // 在dst上画出装甲板的轮廓,粉色,线宽2 
        line(drc, pt[0], pt[1], CV_RGB(255, 0, 255), 2, 8, 0);
        line(drc, pt[1], pt[2], CV_RGB(255, 0, 255), 2, 8, 0);
        line(drc, pt[2], pt[3], CV_RGB(255, 0, 255), 2, 8, 0);
        line(drc, pt[3], pt[0], CV_RGB(255, 0, 255), 2, 8, 0);
    }
}