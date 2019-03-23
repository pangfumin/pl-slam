/*****************************************************************************
**      Stereo VO and SLAM by combining point and line segment features     **
******************************************************************************
**                                                                          **
**  Copyright(c) 2016-2018, Ruben Gomez-Ojeda, University of Malaga         **
**  Copyright(c) 2016-2018, David Zuñiga-Noël, University of Malaga         **
**  Copyright(c) 2016-2018, MAPIR group, University of Malaga               **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify    **
**  it under the terms of the GNU General Public License (version 3) as     **
**  published by the Free Software Foundation.                              **
**                                                                          **
**  This program is distributed in the hope that it will be useful, but     **
**  WITHOUT ANY WARRANTY; without even the implied warranty of              **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            **
**  GNU General Public License for more details.                            **
**                                                                          **
**  You should have received a copy of the GNU General Public License       **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.   **
**                                                                          **
*****************************************************************************/


#include <stereoFrame.h>
#include <stereoFrameHandler.h>
#include <boost/filesystem.hpp>
#include "file-system-tools.h"

#include <mapFeatures.h>
#include <mapHandler.h>

#include <dataset.h>
#include <timer.h>

using namespace StVO;
using namespace PLSLAM;

int main(int argc, char **argv)
{

    // read inputs
    if (argc < 4) {
        std::cerr << "Usage: ./slam config.yaml data_config.yaml dataset_path" << std::endl;
        return 0;
    }

    string dataset_name, config_file, dataset_config_file;

    config_file = std::string(argv[1]);
    dataset_config_file = std::string(argv[2]);
    dataset_name = std::string(argv[3]);
    std::cout << "config file        : " << config_file << std::endl;
    std::cout << "dataset config file: " << dataset_config_file << std::endl;
    std::cout << "dataset_name       : " << dataset_name << std::endl;

    std::string image_left_path = dataset_name + "/cam0/data/";
    std::string image_right_path = dataset_name + "/cam1/data/";
    std::vector<string> image_left_vec, image_right_vec;

    // load image
    common::getAllFilesInFolder(image_left_path, &image_left_vec);
    std::cout<<"image0_list: " << image_left_vec.size() << std::endl;
    std::sort(image_left_vec.begin(),image_left_vec.end(), [](std::string a, std::string b) {
        return !common::compareNumericPartsOfStrings(a,b);
    });

    common::getAllFilesInFolder(image_right_path, &image_right_vec);
    std::cout<<"image1_list: " << image_right_vec.size() << std::endl;
    std::sort(image_right_vec.begin(),image_right_vec.end(), [](std::string a, std::string b) {
        return !common::compareNumericPartsOfStrings(a,b);
    });

    int total_image_num = image_left_vec.size();


    // load config
     SlamConfig::loadFromFile(config_file);
    if (SlamConfig::hasPoints() &&
            (!boost::filesystem::exists(SlamConfig::dbowVocP()) || !boost::filesystem::is_regular_file(SlamConfig::dbowVocP()))) {
        cout << "Invalid vocabulary for points" << endl;
        return -1;
    }

    if (SlamConfig::hasLines() &&
            (!boost::filesystem::exists(SlamConfig::dbowVocL()) || !boost::filesystem::is_regular_file(SlamConfig::dbowVocL()))) {
        cout << "Invalid vocabulary for lines" << endl;
        return -1;
    }

    cout << endl << "Initializing PL-SLAM...." << flush;

    PinholeStereoCamera*  cam_pin = new PinholeStereoCamera(dataset_config_file);

    // create scene
//    string scene_cfg_name = "../config/scene_config_indoor.ini";
//    slamScene scene(scene_cfg_name);
    Matrix4d Tcw, Tfw = Matrix4d::Identity();
    Tcw = Matrix4d::Identity();
//    scene.setStereoCalibration( cam_pin->getK(), cam_pin->getB() );
//    scene.initializeScene(Tfw);



    // create PLSLAM object
    PLSLAM::MapHandler* map = new PLSLAM::MapHandler(cam_pin);

    cout << " ... done. " << endl;

    Timer timer;

    // initialize and run PL-StVO
    int frame_counter = 0;
    StereoFrameHandler* StVO = new StereoFrameHandler(cam_pin);
    Mat img_l, img_r;

    while (frame_counter < total_image_num)
    {
        img_l = cv::imread(image_left_vec[frame_counter],CV_LOAD_IMAGE_UNCHANGED);
        img_r = cv::imread(image_right_vec[frame_counter],CV_LOAD_IMAGE_UNCHANGED);
//        cv::imshow("left", img_l);
//        cv::imshow("right", img_r);
//        cv::waitKey(30);

        if( frame_counter == 0 ) // initialize
        {
            StVO->initialize(img_l,img_r,0);
            PLSLAM::KeyFrame* kf = new PLSLAM::KeyFrame( StVO->prev_frame, 0 );
            map->initialize( kf );
            // update scene
//            scene.initViewports( img_l.cols, img_r.rows );
//            scene.setImage(StVO->prev_frame->plotStereoFrame());
//            scene.updateSceneSafe( map );
        }
        else // run
        {
            // PL-StVO
            timer.start();
            StVO->insertStereoPair( img_l, img_r, frame_counter );
            StVO->optimizePose();
            double t1 = timer.stop(); //ms
            cout << "------------------------------------------   Frame #" << frame_counter
                 << "   ----------------------------------------" << endl;
            cout << endl << "VO Runtime: " << t1 << endl;

            // check if a new keyframe is needed
            if( StVO->needNewKF() )
            {
                cout <<         "#KeyFrame:     " << map->max_kf_idx + 1;
                cout << endl << "#Points:       " << map->map_points.size();
                cout << endl << "#Segments:     " << map->map_lines.size();
                cout << endl << endl;

                // grab StF and update KF in StVO (the StVO thread can continue after this point)
                PLSLAM::KeyFrame* curr_kf = new PLSLAM::KeyFrame( StVO->curr_frame );
                // update KF in StVO
                StVO->currFrameIsKF();
                map->addKeyFrame( curr_kf );
                // update scene
//                scene.setImage(StVO->curr_frame->plotStereoFrame());
//                scene.updateSceneSafe( map );
            }
            else
            {
//                scene.setImage(StVO->curr_frame->plotStereoFrame());
//                scene.setPose( StVO->curr_frame->DT );
//                scene.updateScene();
            }

            cv::Mat image = StVO->curr_frame->plotStereoFrame();
            cv::imshow("image", image);
            cv::waitKey(2);
            // update StVO
            StVO->updateFrame();
        }



        frame_counter++;
    }

    // finish SLAM
    map->finishSLAM();
//    scene.updateScene( map );

    // perform GBA
    cout << endl << "Performing Global Bundle Adjustment..." ;
    map->globalBundleAdjustment();
    cout << " ... done." << endl;
//    scene.updateSceneGraphs( map );

    // wait until the scene is closed
//    while( scene.isOpen() );

    return 0;
}


