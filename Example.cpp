/*=========================================================================
 *
 *  Copyright David Doria 2011 daviddoria@gmail.com
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "KMeansClustering.h"

#include <iostream>

KMeansClustering::VectorOfPoints GenerateData();

int main(int, char *[])
{
  KMeansClustering::VectorOfPoints points = GenerateData();
  KMeansClustering kmeans;
  kmeans.SetK(2);
  kmeans.SetPoints(points);
  kmeans.SetInitMethod(KMeansClustering::KMEANSPP);
  //kmeans.SetRandom(false); // for repeatable results
  kmeans.SetRandom(true); // for real, random results
  kmeans.Cluster();

  std::vector<unsigned int> labels = kmeans.GetLabels();

  std::cout << "Resulting cluster ids:" << std::endl;
  for(unsigned int i = 0; i < labels.size(); ++i)
  {
    std::cout << labels[i] << std::endl;
  }
  
  return EXIT_SUCCESS;
}

KMeansClustering::VectorOfPoints GenerateData()
{
  KMeansClustering::VectorOfPoints points;

  Eigen::VectorXf p = Eigen::VectorXf::Zero(2);
  p[0] = 10; p[1] = 10;
  points.push_back(p);
  p[0] = 10.1; p[1] = 10.1;
  points.push_back(p);
  p[0] = 10.2; p[1] = 10.2;
  points.push_back(p);

  p[0] = 5; p[1] = 5;
  points.push_back(p);
  p[0] = 5.1; p[1] = 5.1;
  points.push_back(p);
  p[0] = 5.2; p[1] = 5.2;
  points.push_back(p);
  
  return points;
}
