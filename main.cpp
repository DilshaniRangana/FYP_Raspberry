#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <ctime>
#include <omp.h>
#include <map>


using namespace std;
using namespace cv;

static map<string, int> BGRValues;
static int Person_Color[2][3] = { {224,224,160},{ 32,32,32 } };
static int rangeGap = 3;


void addto_map(int B, int G, int R)
{

	string key = to_string(B) + ":" + to_string(G) + ":" + to_string(R);
	if (!BGRValues.empty())
	{

		if (BGRValues.find(key)== BGRValues.end())
		{
			BGRValues.insert(pair<string, int>(key, 1));
		}else
		{
			int value = BGRValues.find(key)->second;
			value++;
			BGRValues.find(key)->second = value;
		}
	}
	else
	{
		BGRValues.insert(pair<string,int>(key,1));

	}

}

int findMax(int color)
{
	if (color+ rangeGap <= 255)
	{
		color += rangeGap;

	}
	else {
		color = 255;
	}
	//cout << " max color " << color << endl;
	return color;

}

int findMin(int color)
{
	if (color- rangeGap >=0)
	{
		color -= rangeGap;

	}
	else {
		color = 0;
	}

	//cout << " min color " << color<< endl;
	return color;
}

void colorReduce(cv::Mat& image, int div = 64)
{
	int nl = image.rows;                    // number of lines
	int nc = image.cols * image.channels(); // number of elements per line

	for (int j = 0; j < nl; j++)
	{
		// get the address of row j
		uchar* data = image.ptr<uchar>(j);

		for (int i = 0; i < nc; i++)
		{
			// process each pixel
			data[i] = data[i] / div * div + div / 2;
		}
	}
}

void detectContours(Mat contur)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat gray;
	int thresh = 100;
	int max_thresh = 255;

	cvtColor(contur, gray, COLOR_BGR2GRAY);
	Canny(gray, gray, thresh, thresh * 2, 3);
	findContours(gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	//imshow("Contours ", gray);
}




int main()
{

    int keyboard = 0;
	VideoCapture cap("/media/pi/MULTIBOOT/people.mp4");
	//VideoCapture cap(0);
	Mat image,a,b,c;
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    vector<Rect> found, found_filtered;

    setNumThreads(8);
    cout << getNumberOfCPUs() <<"---"<<getNumThreads()<<endl;

    while ((char)keyboard != 'q' && (char)keyboard != 27) {
		if (!cap.isOpened())
		{
			exit(EXIT_FAILURE);
		}
		cap >> c;
		if (c.empty())
		{
			exit(EXIT_FAILURE);

		}
		resize(c, a, Size(), 0.5, 0.5, INTER_CUBIC);
		b = a.clone();
		/////////use hog descriptor /////////////
		const clock_t beginTime_1 = clock();

		hog.detectMultiScale(a, found, 0, Size(4, 4), Size(16, 16), 1.05, 2);

		cout << "time for HOG " << float(clock() - beginTime_1) / CLOCKS_PER_SEC << endl;

		size_t i, j;
	#pragma omp parallel for
		for (i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}
		//vector<Mat> people;


		for (i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];

			if (r.y > 0 && r.x > 0)
			{

				rectangle(b, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
			}

            imshow("Hog ",b);
        //    imwrite("original.jpg",b);
            waitKey(1);

			cv::Mat result; // segmentation result (4 possible values)
			cv::Mat bgModel, fgModel;

			cv::grabCut(a,result,r,bgModel, fgModel,1,cv::GC_INIT_WITH_RECT);

			cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
			// Generate output image
			cv::Mat foreground(a.size(), CV_8UC3, cv::Scalar(255, 255, 255));
			a.copyTo(foreground, result);
			//cv::imshow("Segmented Image", foreground);


			////////secondly added part////////////////////////////////////////////////////////////////////////////////////

//			cout << "time for grabcut " << float(clock() - beginTime) / CLOCKS_PER_SEC << endl;

			// extract the features
			Mat crop;

			if (0<=r.x && 0<= r.width && r.x +r.width <=foreground.cols && 0<=r.y && 0<=r.height && r.y +r.height <=foreground.rows )
			{
				crop = foreground(r);
			}
			else
			{
				crop = foreground;
			}


			imshow("Cropped image", crop);

	//imwrite("grab_cut.jpg",crop);



			//BGR values

			medianBlur(crop, crop, 5);

			//clear the static map
			if (!BGRValues.empty())
			{
				BGRValues.clear();

			}


			colorReduce(crop);

			//clear background

			//create a 2d array
			int **boundry = new int*[crop.rows];
			for (int i = 0; i < crop.rows; i++)
			{
				boundry[i] = new int[2];

			}


			// get color values
			for (int i = 0; i < crop.rows; i++)
			{
				int left = 0, right = 0, lvalidate = 0, rvalidate = 0;vector<int> v;
				for (int j = 0, k = crop.cols - 1; j <= crop.cols / 2; j++, k--)
				{

					int B = crop.at<cv::Vec3b>(i, j)[0];
					int G = crop.at<cv::Vec3b>(i, j)[1];
					int R = crop.at<cv::Vec3b>(i, j)[2];
					int checkColor = 224;

					if ((B!= checkColor && G!= checkColor && R!= checkColor) && lvalidate==0)
					{
						left = j;
						lvalidate = 1;
						addto_map(B, G, R);
						boundry[i][0] = j;


					}else if(lvalidate==1)
					{
						addto_map(B, G, R);

					}

					int Br = crop.at<cv::Vec3b>(i, k)[0];
					int Gr= crop.at<cv::Vec3b>(i, k)[1];
					int Rr = crop.at<cv::Vec3b>(i, k)[2];

					if ((Br != checkColor && Gr != checkColor && Rr != checkColor) && rvalidate == 0)
					{
						right = k;
						rvalidate = 1;
						addto_map(Br, Gr, Rr);
						boundry[i][1] = k;
					}
					else if (rvalidate == 1)
					{
						addto_map(Br, Gr, Rr);

					}

				}
				if (lvalidate == 0 && rvalidate == 1)
				{

					boundry[i][0] = right;

				}
				if (rvalidate == 0 && lvalidate == 1)
				{
					boundry[i][1] = left;

				}

			}

			//find width to height ratio;

			int totwidth = 0, num=0;
			int h1 = 0, h2 = 0;
			for (int i = 0; i < crop.rows; i++)
			{


				if (boundry[i][0]>= 0 || boundry[i][1]<=crop.cols)
				{
					h2 = i;
					if (h1==0)
					{
						h1 = i;
					}
					totwidth += (boundry[i][1] - boundry[i][0]);
					num++;

				}
			}

			int avgWidth = totwidth / num;
			int height = h2 - h1;
			float w2h = (float)avgWidth / (float)height;

			cout << avgWidth << ":" << height << ":" << w2h << endl;


			//end of find width to height ratio;


			//find maximum colors
			int max1 = 0, max2 = 1, max3 = 0,max4=0;
			string element[4];

			for (int i = 0; i < 4; i++)
			{
				element[i] = "_:_:_";
			}

			for (auto elem : BGRValues)
			{
				int value = elem.second;
				string index = elem.first;
				if (max1 < value)
				{
					max4 = max3;
					max3 = max2;
					max2 = max1;
					max1 = value;

					element[3] = element[2];
					element[2] = element[1];
					element[1] = element[0];
					element[0]  = index;

				}
				else if (max2 < value)
				{
					max4 = max3;
					max3 = max2;
					max2 = value;

					element[3] = element[2];
					element[2] = element[1];
					element[1] = index;

				}
				else if (max3< value)
				{
					max4 = max3;
					max3 = value;

					element[3] = element[2];
					element[2] = index;

				}
				else if (max4< value)
				{
					max4 = value;
					element[3] = index;

				}


			}

			cout << " max 1 " << element[0] << " " << max1 << " max 2 " << element[1] << " " << max2 << " max 3 " << element[2] << " " << max3 << " max 4 " << element[3] << endl;



			//detect contours



			int maxColors[4][3];


			//convert colors to rgb values
			int control = 0;
			for (int i = 0; i < 4; i++)
			{
				istringstream iss(element[i]);
				string s;
				int j = 0;
				while (getline(iss, s, ':')) {
                 /*   if (s.compare("_") ==0)
					{
						break;
						control = i;
					} */

					maxColors[i][j]= atoi(s.c_str());
					j++;

				}
			}

			int colorRange[4][3][2];

			bool ok1 = false;
			bool ok2 = false;
			for (int i = 0; i < 4; i++)
			{

				if ((maxColors[i][0] == Person_Color[0][0] && maxColors[i][1] == Person_Color[0][1] && maxColors[i][2] == Person_Color[0][2]) || (maxColors[i][0] == Person_Color[1][0] && maxColors[i][1] == Person_Color[1][1] && maxColors[i][2] == Person_Color[1][2]))
				{

					if (ok1)
					{

						ok2 = true;
					}
					else
					{
						ok1 = true;
						if (Person_Color[0][0] == Person_Color[1][0] && Person_Color[0][1] == Person_Color[1][1] && Person_Color[0][2] == Person_Color[1][2])
						{

							ok2 = true;

						}
					}
				}

			}



			cout << "image content " << ok1 << " " << ok2 << endl;

			if (ok1 && ok2)
			{

				Mat bin1[3];
				Rect rectangles[3];

				//process single image

				for (int i = 0; i < 2; i++)
				{

					//find the range for one color
					for (int j = 0; j <3; j++)
					{
						int color = Person_Color[i][j];
						colorRange[i][j][0] = findMin(color);
						colorRange[i][j][1] = findMax(color);

					}


					//extract the color binary
					inRange(crop, Scalar(colorRange[i][0][0], colorRange[i][1][0], colorRange[i][2][0]), Scalar(colorRange[i][0][1], colorRange[i][1][1], colorRange[i][2][1]), bin1[i]);


					imshow("one color ", bin1[i]);
					//imwrite("one_color.jpg",bin1[i]);

					vector<vector<Point> > contours;
					vector<Vec4i> hierarchy;

					//
					findContours(bin1[i], contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

					/// Approximate contours to polygons + get bounding rects and circles
					vector<vector<Point> > contours_poly(contours.size());
					vector<Rect> boundRect(contours.size());
					vector<Point2f>center(contours.size());
					vector<float>radius(contours.size());

					Rect maximum_rect;
					int size = 0;
					int index;
					Mat drawing = Mat::zeros(bin1[i].size(), CV_8UC3);
					for (int k = 0; k < contours.size(); k++)
					{
						approxPolyDP(Mat(contours[k]), contours_poly[k], 3, true);
						boundRect[k] = boundingRect(Mat(contours_poly[k]));
						if (size< boundRect[k].area())
						{
							size = boundRect[k].area();
							index = k;
						}

						//	cout << boundRect[k].area() << endl;



					}
					//draw the rectangle
					rectangles[i] = boundRect[index];
					rectangle(drawing, boundRect[index].tl(), boundRect[index].br(), Scalar(0, 255, 0), 2, 8, 0);
					imshow("rect ", drawing);
					//imwrite("rect.jpg", drawing);

					waitKey(0);






					cout << Person_Color[i][2] << "," << Person_Color[i][1] << "," << Person_Color[i][0] << endl;




				}

				int xCordinates[3];
				int yCordinates[3];
				int maxX = 0, xind;
				int maxY = 0, yind;

				for (int i = 0; i < sizeof(rectangles)/sizeof(Rect); i++)
				{
					xCordinates[i] = rectangles[i].tl().x;
					if (maxX < rectangles[i].tl().x)
					{
						maxX = rectangles[i].tl().x;
						xind = i;
					}

					if (maxY < rectangles[i].tl().y)
					{
						maxY = rectangles[i].tl().y;
						yind = i;
					}


					yCordinates[i] = rectangles[i].tl().y;
				}

				//if upper color = lower color code goes here





				//if upper color = lower color code ends here



				//upperbody color
				int Lb = Person_Color[yind][0];
				int Lg = Person_Color[yind][1];
				int Lr = Person_Color[yind][2];

				cout << "lower body color " << Lb << "," << Lg << "," << Lr << endl;
				Mat img(500, 500, CV_8UC3);
				img = cv::Scalar(Lb, Lg, Lr);
			//	imshow("Lower body ", img);
				//imwrite("Lower_body.jpg ", img);

				int upper = 2;
				if (yind == 0)
				{
					upper = 1;
				}
				else
				{
					upper = 0;
				}

				 Lb = Person_Color[upper][0];
				 Lg = Person_Color[upper][1];
				 Lr = Person_Color[upper][2];


				Mat img1(500, 500, CV_8UC3);
				img1 = cv::Scalar(Lb, Lg, Lr);
			//	imshow("Upper body ", img1);
				//imwrite("Upper_body.jpg ", img1);



			}


			//end of detecting contours

			//end of BGR values




			//end of extracting




			keyboard = waitKey(1);

		}

    }

return 0;
}
