#include <iostream>
#include <cstdint>
#include <cstring>
#include <opencv2/opencv.hpp>
using namespace std;

int main(int argc, char *argv[])
{
    cv::Mat m(10, 512, CV_32F);
    std::vector<float> arr(512,1.5);
    for ( int i = 0; i < 10; i++)
    {
        auto p = m.data + sizeof(float)/sizeof(uchar)*i*512;
        memcpy(p,&arr[0], sizeof(float)*512);
    }
    std::cout << m << std::endl;
    cout << "Hello World!" << endl;
    return 0;
}
