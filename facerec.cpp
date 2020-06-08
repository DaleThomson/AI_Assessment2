#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

// g++ -std=c++1z 1_simple_facerec_eigenfaces.cpp -lopencv_face -lopencv_core -lopencv_imgcodecs -lstdc++fs

int main(int argc, char *argv[])
{
  std::vector<cv::Mat> images;
  std::vector<int>     labels;
  cv::Mat frame;
  cv::Mat crop;
  cv::Mat cropGrey;
  int predictedLabel = NULL;
  double fps = 30;
  const char win_name[] = "Video Feed";

  cv::VideoCapture vid_in(0);   // argument is the camera id
  if (!vid_in.isOpened()) {
	  std::cout << "error: Camera 0 could not be opened for capture.\n";
	  return -1;
  }

  // Iterate through all subdirectories, looking for .pgm files
  fs::path p(argc > 1 ? argv[1] : "../att_faces");
  for (const auto &entry : fs::recursive_directory_iterator{ p }) {
    if (fs::is_regular_file(entry.status())) { // Was once always (wrongly) false in VS
      if (entry.path().extension() == ".pgm") {
        std::string str = entry.path().parent_path().stem().string(); // s26 s27 etc.
        int label = atoi(str.c_str() + 1); // s1 -> 1 (pointer arithmetic)
        images.push_back(cv::imread(entry.path().string().c_str(), cv::IMREAD_GRAYSCALE));
        labels.push_back(label);
      }
    }
  }
  std::cout << " training...";

  cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
  model->train(images, labels);
  cv::namedWindow(win_name);
  while(1)
  {
	  vid_in >> frame;
	  cv::putText(frame, "Predicted Sample = " + std::to_string(predictedLabel), cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
	  cv::rectangle(frame, cv::Point(275, 200), cv::Point(367, 312), cv::Scalar(0, 255, 0),1, 8);
	  imshow(win_name, frame);
	  crop = frame(cv::Rect(275, 200, 92, 112));
	  cv::cvtColor(crop, cropGrey, CV_RGB2GRAY);
	  predictedLabel = model->predict(cropGrey);
	  int code = cv::waitKeyEx(1000 / fps); // how long to wait for a key (msecs)
	  if (code == 27) // escape. See http://www.asciitable.com/
		  break;
  }
  vid_in.release();
  return 0;
}