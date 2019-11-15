//显著性计算
//参考论文：Saliency Detection: A Spectral Residual Approach

#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <string>
 
using namespace cv;
using namespace std;

//傅里叶正变换
void fft2(cv::Mat &src, cv::Mat &dst);


void fft2(cv::Mat &src, cv::Mat &dst)
{   
    cv::Mat image_Re(src.rows,src.cols, CV_64FC1);
    cv::Mat Fourier(src.rows,src.cols, CV_64FC2);

    //实部的值初始设为源图像，虚部的值初始设为0
    // Real part conversion from u8 to 64f (double)
    src.convertTo(image_Re, CV_64FC1, 1, 0);
    // Imaginary part (zeros)
    cv::Mat image_Im = Mat::zeros(src.size(),CV_64FC1);
    
    // Join real and imaginary parts and stock them in Fourier image
    Mat planes[] = {image_Re,image_Im};

	merge(planes, 2, Fourier);
    cv::dft(Fourier, dst);
    
}

int main( int argc, char** argv )
{
    
    string imagePath = argv[1];   

    double minNum = 0, maxNum = 0, scale, shift;
    int i, j,nRow, nCol;

    cv::Mat src;
    src=cv::imread(imagePath,0);
    
    int width=src.cols;
    int height=src.rows;
    //注意Fourier是一个两通道的图像，一个通道为实部，一个为虚部
    
    cv::Mat Fourier(height, width,CV_64FC2);
    cv::Mat Inverse(height, width,CV_64FC2);
    cv::Mat ImageRe(height, width,CV_64FC1);
    cv::Mat ImageIm(height, width,CV_64FC1);
    cv::Mat LogAmplitude(height, width,CV_64FC1);
    cv::Mat Sine(height, width,CV_64FC1);
    cv::Mat Cosine(height, width,CV_64FC1);
    cv::Mat Residual(height, width,CV_64FC1);
    cv::Mat tmp1(height, width,CV_64FC1);
    cv::Mat tmp2(height, width,CV_64FC1);
    cv::Mat tmp3(height, width,CV_64FC1);
    cv::Mat Saliency(height, width,CV_64FC1);
    
    //归一化
    scale = 1.0/255.0;
    src.convertTo(tmp1, CV_64FC1, scale, 0);
    
    fft2(tmp1, Fourier);
    
    std::vector<cv::Mat> chls;
    cv::split(Fourier, chls);
    ImageRe=chls.at(0);
    ImageIm=chls.at(1);
    
    cv::magnitude(ImageRe,ImageIm,tmp3);
    cv::log(tmp3, LogAmplitude);
    
    cv::divide(ImageIm, tmp3, Sine);
    cv::divide(ImageRe, tmp3, Cosine);
    
    cv::blur(LogAmplitude, tmp3, Size(3, 3), Point(-1,-1));
    Residual = LogAmplitude-tmp3;
    cv::exp(Residual, Residual);
    tmp1=Residual.mul(Cosine);
    tmp2=Residual.mul(Sine);
    
    Mat Phase[] = {tmp1,tmp2};

	merge(Phase, 2, Fourier);
	
	cv::dft(Fourier, Inverse, DFT_INVERSE);
    
    std::vector<cv::Mat> Invs;
    cv::split(Inverse, Invs);
    tmp1=Invs.at(0);
    tmp2=Invs.at(1);
    tmp3 = tmp1.mul(tmp1) + tmp2.mul(tmp2);
    cv::GaussianBlur(tmp3, tmp3, cv::Size(7, 7), 0, 0, BORDER_DEFAULT);
    
    cv::minMaxLoc(tmp3, &minNum, &maxNum);
    //std::cout<<minNum<<std::endl;
    //std::cout<<maxNum<<std::endl;
    scale = 255/(maxNum - minNum);
    shift = -minNum * scale;
    
    tmp3.convertTo(Saliency, CV_64FC1, scale, shift);
    
    cv::imshow("src",src);
    cv::imshow("InvDFT",tmp3);
    cv::imshow("Saliency",Saliency);
    cv::waitKey(0);
    
    return 0;
}


