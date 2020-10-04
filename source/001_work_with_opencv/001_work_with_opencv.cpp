
#include <stdio.h>
#include <Halide.h>
#include <halide_image_io.h>
#include <opencv2/opencv.hpp>


int main(int argc, char* argv[]) {
#ifdef HAVE_OPENCV
    cv::Mat image = cv::imread("images/rgb.png");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::imshow("in", image);
    cv::waitKey(10);
    Halide::Buffer<uint8_t> input(Halide::Buffer<uint8_t>::make_interleaved(image.data, image.cols, image.rows, image.channels()));

    // If the OpenCV matrix has padding between the rows, the longer form is
    // a halide_dimension_t is the min coordinate, the extent, and then the stride/step in that dimension.
    //halide_dimension_t shape[3] = { {0, image.cols, image.step1(1)},
    //                           {0, image.rows, image.step1(0)},
    //                           {0, image.channels(), 1}
    //};
    //Halide::Buffer<uint8_t> buffer(image.data, 3, shape);
#else
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("images/rgb.png");
#endif
    Halide::Func brighter;
    Halide::Var x, y, c;
    Halide::Expr value = input(x, y, c);
    value = Halide::cast<float>(value);
    value = value * 1.5f;
    value = Halide::min(value, 255.0f);
    value = Halide::cast<uint8_t>(value);
    brighter(x, y, c) = value;

    Halide::Buffer<uint8_t> output =
        brighter.realize(input.width(), input.height(), input.channels());

#ifdef HAVE_OPENCV
    cv::Mat image_out = cv::Mat::zeros(output.height(), output.width(), CV_8UC3);
    for (int i = 0; i < output.height(); ++i) {
        for (int j = 0; j < output.width(); ++j) {
            for (int n = 0; n < output.channels(); ++n) {
                image_out.at<uchar>(i, j*output.channels()+n) = (uchar)output(j, i, n);
            }
        }
    }
    cv::imshow("out", image_out);
    cv::waitKey(10);
#else
    Halide::Tools::save_image(output, "brighter.png");
#endif

    return 0;
}